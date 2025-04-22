from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter, Instruction
import numpy as np
import networkx as nx
from scipy.linalg import block_diag
from qiskit_prog_qaoa.utils.logging import get_logger

logger = get_logger(__name__)


def is_equal_to(num_qubits: int, value: int) -> QuantumCircuit:
    """
    Creates a circuit that checks if a register encodes a value in binary and flips a flag.
    num_qubits: size of register to check
    value: value to check
    """
    circ = QuantumCircuit(num_qubits + 1)
    circ.mcx(list(range(num_qubits-1,-1,-1)), num_qubits, ctrl_state=value)
    return circ


def controlled_copy_with_swap(num_qubits: int, K: int) -> QuantumCircuit:
    """
    Creates a controlled copy circuit that also shuffles all the registers in the copy list forward one place.
    num_qubits: size of register to be copied
    K: number of registers to track copies in
    """
    # Qubits: c_copy_flag, to_be_copied, K * (reg_to_be_copied_into)
    circ = QuantumCircuit(1 + (K+1) * num_qubits)
    
    # For each qubit i in final copy register:
    for i in range(num_qubits):
        # Swap down to ith position in the first copy register, conditioned on the c_copy_flag
        for idx in range(K * num_qubits + i + 1, 1+num_qubits+i, -1):
            circ.cswap(0, idx, idx-1)
            
    # Copy the to_be_copied register into the first copy register, conditioned on the c_copy_flag
    for i in range(num_qubits):
        circ.ccx(0, i + 1, num_qubits + i + 1)
    return circ


def compute_next_nodes(
        circuit: QuantumCircuit, registers: dict, j: int, n: int, K: int, T: int
) -> QuantumCircuit:
    """
    Appends a compute_next_nodes subroutine to a circuit, which initialises registers .
    For t in 0..T-2:
        Checks if the t th solution register encodes j and flips a flag if so.
        If the flag is set, shuffles the next_node registers forward one place and copies the (t+1)th solution register to the next_node registers.
        Resets the flag.
    """
    ceil_log_n2 = int(np.ceil(np.log2(n+2)))
    cc_circ = controlled_copy_with_swap(ceil_log_n2, K)
    is_equal_circ = is_equal_to(ceil_log_n2, j)
    for t in range(T-1):
        # circuit.barrier(label=f'is_equal c_{t}, {j}')
        circuit.compose(
            is_equal_circ, 
            list(range(circuit.find_bit(registers[f'solution_{t}'][0]).index, circuit.find_bit(registers[f'solution_{t}'][-1]).index + 1)) \
                + [circuit.find_bit(registers['flag'][0]).index],
            inplace=True
        )

        # circuit.barrier(label=f'c_copy c_{t+1} -> next node list')
        flag = circuit.find_bit(registers['flag'][0]).index
        to_copy = list(range(
            circuit.find_bit(registers[f'solution_{t+1}'][0]).index, circuit.find_bit(registers[f'solution_{t+1}'][-1]).index + 1
        ))
        copy_registers = list(range(
            circuit.find_bit(registers['next_node_0'][0]).index, circuit.find_bit(registers[f'next_node_{K-1}'][-1]).index + 1
        ))

        circuit.compose(
            cc_circ, 
            [flag] + to_copy + copy_registers,
            inplace=True
        )

        # circuit.barrier(label=f'uncompute is_equal c_{t}, {j}')
        circuit.compose(
            is_equal_circ, 
            list(range(circuit.find_bit(registers[f'solution_{t}'][0]).index, circuit.find_bit(registers[f'solution_{t}'][-1]).index + 1)) \
                + [circuit.find_bit(registers['flag'][0]).index],
            inplace=True
        )
        # circuit.barrier()
    return circuit


def uncompute_next_nodes(circuit: QuantumCircuit, registers, j, n, K, T):
    ceil_log_n2 = int(np.ceil(np.log2(n+2)))
    cc_circ = controlled_copy_with_swap(ceil_log_n2, K)
    is_equal_circ = is_equal_to(ceil_log_n2, j)
    for t in range(T-2, -1, -1):
        # circuit.barrier(label=f'is_equal c_{t}, {j}')
        circuit.compose(
            is_equal_circ, 
            list(range(circuit.find_bit(registers[f'solution_{t}'][0]).index, circuit.find_bit(registers[f'solution_{t}'][-1]).index + 1)) \
                + [circuit.find_bit(registers['flag'][0]).index],
            inplace=True
        )

        # circuit.barrier(label=f'c_copy c_{t+1} -> next node list')
        flag = circuit.find_bit(registers['flag'][0]).index
        to_copy = list(range(
            circuit.find_bit(registers[f'solution_{t+1}'][0]).index, circuit.find_bit(registers[f'solution_{t+1}'][-1]).index + 1
        ))
        copy_registers = list(range(
            circuit.find_bit(registers['next_node_0'][0]).index, circuit.find_bit(registers[f'next_node_{K-1}'][-1]).index + 1
        ))

        circuit.compose(
            cc_circ.reverse_ops(), 
            [flag] + to_copy + copy_registers,
            inplace=True
        )

        # circuit.barrier(label=f'uncompute is_equal c_{t}, {j}')
        circuit.compose(
            is_equal_circ, 
            list(range(circuit.find_bit(registers[f'solution_{t}'][0]).index, circuit.find_bit(registers[f'solution_{t}'][-1]).index + 1)) \
                + [circuit.find_bit(registers['flag'][0]).index],
            inplace=True
        )
        # circuit.barrier()
    return circuit


def penalise_graph_steps(
        circuit: QuantumCircuit, registers: dict, i: int, parameter: Parameter, graph: nx.Graph, n: int, K:int
) -> QuantumCircuit:
    """
    Appends a penalise_graph_steps subroutine to a circuit, which penalises any step from node i to a node not adjacent to i.
    For each node j in 1..n not adjacent to i:
        For each possible number of visits k:
            Checks if the kth next node register is equal to j, and flips a flag if so.
            Applies a phase to the flag qubit proportional to the parameter.
            Resets the flag.
    """
    ceil_log_n2 = int(np.ceil(np.log2(n+2)))
    nodes = list(graph.nodes)
    for j in range(1, n+1):
        if (nodes[i-1], nodes[j-1]) not in graph.edges:
            is_equal_circ = is_equal_to(ceil_log_n2, j)
            # circuit.barrier(label=f'penalty for {nodes[i-1], nodes[j-1]}')
            for k in range(K):
                circuit.compose(
                    is_equal_circ,
                    list(range(
                        circuit.find_bit(registers[f'next_node_{k}'][0]).index, circuit.find_bit(registers[f'next_node_{k}'][-1]).index + 1
                    )) + [circuit.find_bit(registers['flag'][0]).index],
                    inplace=True
                )
                circuit.p(
                    parameter, 
                    circuit.find_bit(registers['flag'][0]).index
                )
                circuit.compose(
                    is_equal_circ,
                    list(range(
                        circuit.find_bit(registers[f'next_node_{k}'][0]).index, circuit.find_bit(registers[f'next_node_{k}'][-1]).index + 1
                    )) + [circuit.find_bit(registers['flag'][0]).index],
                    inplace=True
                )
        # circuit.barrier()
    return circuit


def penalise_graph_end_steps(
    circuit: QuantumCircuit, registers: dict, parameter: Parameter, n: int, K: int
) -> QuantumCircuit:
    """
    Appends a penalise_graph_end_steps subroutine to a circuit, which penalises any step from the end node to a non-end node.
    For each node j in 1..n:
        For each possible number of visits k:
            Checks if the kth next node register is equal to j, and flips a flag if so.
            Applies a phase to the flag qubit proportional to the parameter.
            Resets the flag.
    """
    ceil_log_n2 = int(np.ceil(np.log2(n+2)))
    # nodes = list(graph.nodes)
    for j in range(1, n+1):
        is_equal_circ = is_equal_to(ceil_log_n2, j)
        # circuit.barrier(label=f'penalty for {nodes[-1], nodes[j-1]}')
        for k in range(K):
            circuit.compose(
                is_equal_circ,
                list(range(
                    circuit.find_bit(registers[f'next_node_{k}'][0]).index, circuit.find_bit(registers[f'next_node_{k}'][-1]).index + 1
                )) + [circuit.find_bit(registers['flag'][0]).index],
                inplace=True
            )
            circuit.p(
                parameter, 
                circuit.find_bit(registers['flag'][0]).index
            )
            circuit.compose(
                is_equal_circ,
                list(range(
                    circuit.find_bit(registers[f'next_node_{k}'][0]).index, circuit.find_bit(registers[f'next_node_{k}'][-1]).index + 1
                )) + [circuit.find_bit(registers['flag'][0]).index],
                inplace=True
            )
    return circuit


def get_constraint_circuit(
        n: int,
        K: int,
        T: int,
        graph: nx.Graph,
        parameter=Parameter('theta_cons'),
        state_prep_circuit: QuantumCircuit | None = None,
) -> QuantumCircuit:
    """
    Prepares a quantum circuit for the constraint function for Tangle Prog-QAOA.
    """
    ceil_log_n2 = int(np.ceil(np.log2(n+2)))
    circuit = QuantumCircuit()

    registers = {f'solution_{t}' : QuantumRegister(ceil_log_n2, name=f'solution_{t}') for t in range(T)}
    registers.update({f'next_node_{k}': QuantumRegister(ceil_log_n2, name=f'next_node_{k}') for k in range(K)})
    registers.update({'flag': QuantumRegister(1, name='flag')})


    for register in registers.values():
        circuit.add_register(register)

    if state_prep_circuit is not None:
        circuit.compose(state_prep_circuit, list(range(T * ceil_log_n2)), inplace=True)

    for j in range(1, n+1):
        circuit = compute_next_nodes(circuit, registers, j, n, K, T)
        # circuit.save_statevector(label=f'after_compute_next_nodes_{j}')
        circuit = penalise_graph_steps(circuit, registers, j, parameter, graph, n, K)
        # circuit.save_statevector(label=f'after_penalise_{j}')
        circuit = uncompute_next_nodes(circuit, registers, j, n, K, T)
        # circuit.save_statevector(label=f'after_uncompute_next_nodes_{j}')

    circuit = compute_next_nodes(circuit, registers, n+1, n, K, T)
    # circuit.save_statevector(label='after_compute_next_nodes_end')
    circuit = penalise_graph_end_steps(circuit, registers, parameter, n, K)
    # circuit.save_statevector(label='after_penalise_end')
    circuit = uncompute_next_nodes(circuit, registers, n+1, n, K, T)
    # circuit.save_statevector(label='after_uncompute_next_nodes_end')
    
    return circuit


def compute_count(circuit: QuantumCircuit, registers: dict, j: int, n: int, K: int, T: int) -> QuantumCircuit:
    """
    Appends a compute_count subroutine to a circuit.
    For each register x_0 ... x_{T-1}, checks if the register encodes j in binary and flips a flag if so.
    Adds 1 to a count register based on the flag.
    Resets the flag.
    """
    ceil_log_n2 = int(np.ceil(np.log2(n+2)))
    ceil_log_K1 = int(np.ceil(np.log2(K+1)))
    is_equal_circ = is_equal_to(ceil_log_n2, j)

    add_one_matrix = np.diag(np.ones(2 ** ceil_log_K1 - 1), -1)
    add_one_matrix[0, -1] = 1

    control_add_one = block_diag(np.eye(add_one_matrix.shape[0]), add_one_matrix)
    for t in range(T):
        circuit.compose(
            is_equal_circ, 
            list(range(circuit.find_bit(registers[f'solution_{t}'][0]).index, circuit.find_bit(registers[f'solution_{t}'][-1]).index + 1)) \
                + [circuit.find_bit(registers['flag'][0]).index],
            inplace=True
        )
        
        # circuit.save_statevector(f'before_c_add_{j}_{t}')
        circuit.unitary(
            control_add_one, 
            list(range(circuit.find_bit(registers['count'][-1]).index, circuit.find_bit(registers['count'][0]).index - 1, -1)) + \
                [circuit.find_bit(registers['flag'][0]).index],
            label='control-add-1'
        )
        # circuit.save_statevector(f'after_c_add_{j}_{t}')
        circuit.compose(
            is_equal_circ, 
            list(range(circuit.find_bit(registers[f'solution_{t}'][0]).index, circuit.find_bit(registers[f'solution_{t}'][-1]).index + 1)) \
                + [circuit.find_bit(registers['flag'][0]).index],
            inplace=True
        )
        
    return circuit


def penalise_count(
        circuit: QuantumCircuit, registers: dict, j: int, parameter: Parameter, graph: nx.Graph, K: int
) -> QuantumCircuit:
    """
    Appends a penalise_count subroutine to a circuit for node j.
    For each possible number of visits to a node 0..K not equal to the weight of node j:
        Checks if count equals that number of visits and flips a flag.
        Applies a phase gate to the flag qubit based on the cost of that number of visits.
        Resets the flag.
    """
    ceil_log_K1 = int(np.ceil(np.log2(K+1)))
    nodes = list(graph.nodes)
    for i in range(K+1):
        if not graph.nodes[nodes[j-1]]["weight"] - i == 0:
            is_equal_circ = is_equal_to(ceil_log_K1, i)

            circuit.compose(
                is_equal_circ,
                list(range(circuit.find_bit(registers['count'][0]).index, circuit.find_bit(registers['count'][-1]).index + 1)) \
                    + [circuit.find_bit(registers['flag'][0]).index],
                inplace=True
            )
            # circuit.save_statevector(label=f'before_p_{j}_{i}')
            circuit.p(parameter * (graph.nodes[nodes[j-1]]["weight"] - i) ** 2, circuit.find_bit(registers['flag'][0]).index)
            # circuit.save_statevector(label=f'after_p_{j}_{i}')

            circuit.compose(
                is_equal_circ,
                list(range(circuit.find_bit(registers['count'][0]).index, circuit.find_bit(registers['count'][-1]).index + 1)) \
                    + [circuit.find_bit(registers['flag'][0]).index],
                inplace=True
            )
    return circuit


def uncompute_count(circuit: QuantumCircuit, registers: dict, j: int, n: int, K: int, T: int) -> QuantumCircuit:
    """
    Appends an uncompute_count subroutine to a circuit.
    For each register x_0 ... x_{T-1}, checks if the register encodes j in binary and flips a flag if so.
    Subtracts 1 to a count register based on the flag.
    Resets the flag.
    """
    ceil_log_n2 = int(np.ceil(np.log2(n+2)))
    ceil_log_K1 = int(np.ceil(np.log2(K+1)))
    is_equal_circ = is_equal_to(ceil_log_n2, j)

    minus_one_matrix = np.diag(np.ones(2 ** ceil_log_K1 - 1), 1)
    minus_one_matrix[-1, 0] = 1
    control_minus_one = block_diag(np.eye(minus_one_matrix.shape[0]), minus_one_matrix)
    for t in range(T):
        circuit.compose(
            is_equal_circ, 
            list(range(circuit.find_bit(registers[f'solution_{t}'][0]).index, circuit.find_bit(registers[f'solution_{t}'][-1]).index + 1)) \
                + [circuit.find_bit(registers['flag'][0]).index],
            inplace=True
        )
        
        circuit.unitary(
            control_minus_one, 
            list(range(circuit.find_bit(registers['count'][-1]).index, circuit.find_bit(registers['count'][0]).index - 1, -1)) + \
                [circuit.find_bit(registers['flag'][0]).index],
            label='control-minus-1'
        )

        circuit.compose(
            is_equal_circ, 
            list(range(circuit.find_bit(registers[f'solution_{t}'][0]).index, circuit.find_bit(registers[f'solution_{t}'][-1]).index + 1)) \
                + [circuit.find_bit(registers['flag'][0]).index],
            inplace=True
        )
    return circuit


def get_objective_circuit(
        n: int,
        K: int,
        T: int,
        graph: nx.Graph,
        parameter=Parameter('theta_obj'),
        state_prep_circuit: QuantumCircuit | None = None, 
) -> QuantumCircuit:
    """
    Prepares a quantum circuit for the objective function for Tangle Prog-QAOA.
    """
    circuit = QuantumCircuit()

    ceil_log_n2 = int(np.ceil(np.log2(n+2)))
    ceil_log_K1 = int(np.ceil(np.log2(K+1)))
    registers = {f'solution_{t}' : QuantumRegister(ceil_log_n2, name=f'solution_{t}') for t in range(T)}
    registers.update({'flag': QuantumRegister(1, name='flag')})
    registers.update({'count': QuantumRegister(ceil_log_K1, name='count')})

    for register in registers.values():
        circuit.add_register(register)

    if state_prep_circuit is not None:
        circuit.compose(state_prep_circuit, list(range(T * ceil_log_n2)), inplace=True)

    for j in range(1, n+1):
        # circuit.save_statevector(label=f'before_count_{j}')
        circuit = compute_count(circuit, registers, j, n, K, T)
        # circuit.save_statevector(label=f'after_count_{j}')
        # circuit.barrier()
        circuit = penalise_count(circuit, registers, j, parameter, graph, K)
        # circuit.save_statevector(label=f'after_penalise_{j}')|
        # circuit.barrier()
        circuit = uncompute_count(circuit, registers, j, n, K, T)
        # circuit.save_statevector(label=f'after_uncount_{j}')
        # circuit.barrier()

    return circuit


def uniform_over_range(num_qubits: int, M: int):
    """
    Returns a circuit that prepares a uniform superposition over |1>,|2>,...,|M> on num_qubits qubits.
    Uses a Hadamard layer if M is a power of 2, else uses the method of Shukla and Vedula.
    """
    if M not in range(2, 2 ** num_qubits + 1):
        raise Exception('Bad M: out of range')
    for i in range(num_qubits):
        if M == 2 ** i:
            logger.info(f'M={M} a power of 2. Use Hadamard circuit.')
            circuit = QuantumCircuit(num_qubits)
            for j in range(i):
                circuit.h(j)
            add_one_matrix = np.diag(np.ones(2 ** num_qubits - 1), -1)
            add_one_matrix[0, -1] = 1
            circuit.unitary(add_one_matrix, list(range(num_qubits)))
            return circuit
    
    circuit = QuantumCircuit(num_qubits)

    M_binary = np.binary_repr(M, num_qubits)
    M_binary = M_binary[::-1]
    ran = np.arange(len(M_binary))
    mask = [M_binary[x] == '1' for x in range(len(M_binary))]
    l = ran[mask]
    
    for i in range(1, len(l)):
        circuit.x(l[i])
    if l[0] > 0:
        for i in range(l[0]):
            circuit.h(i)

    MM = 2 ** l[0]


    circuit.ry(-2 * np.arccos(np.sqrt(MM/M)), l[1])

    for i in range(l[0], l[1]):
        circuit.ch(l[1], i, ctrl_state=0)

    for m in range(1, len(l)-1):
        circuit.cry(
            -2 * np.arccos(np.sqrt(2 ** l[m] / (M - MM) )), 
            l[m], l[m+1], ctrl_state=0
        )
        for i in range(l[m], l[m+1]):
            circuit.ch(l[m+1], i, ctrl_state=0)
        MM += 2 ** l[m]

    add_one_matrix = np.diag(np.ones(2 ** num_qubits - 1), -1)
    add_one_matrix[0, -1] = 1
    circuit.unitary(add_one_matrix, list(range(num_qubits)))
    return circuit


def state_prep(n: int, T: int) -> QuantumCircuit:
    ceil_log_n2 = int(np.ceil(np.log2(n+2)))
    uni = uniform_over_range(ceil_log_n2, n+1)
    circuit = QuantumCircuit(ceil_log_n2 * T)
    for t in range(T):
        circuit.compose(
            uni,
            list(range(t * ceil_log_n2, (t+1)* ceil_log_n2)),
            inplace=True
        )
    return circuit


def get_mixer_operator(n: int, T: int, parameter=Parameter('beta')) -> QuantumCircuit:
    # TODO: use ancillas to reduce depth of mcp?
    num_qubits = int(np.ceil(np.log(n+2))) * T
    state_prep_circuit = state_prep(n, T)
    mixer = QuantumCircuit(state_prep_circuit.num_qubits)
    mixer.compose(
        state_prep_circuit.inverse(),
        range(state_prep_circuit.num_qubits),
        inplace=True
    )
    # mixer.save_statevector('after_prep')
    mixer.x(-1)
    mixer.mcp(-parameter, list(range(num_qubits - 1)), -1, ctrl_state=0)
    mixer.x(-1)
    # mixer.save_statevector('after_phase')
    mixer.compose(
        state_prep_circuit,
        range(state_prep_circuit.num_qubits),
        inplace=True
    )
    # mixer.save_statevector('after_unprep')
    return mixer


def get_phase_operator_instruction(n: int, K: int, T: int, graph: nx.Graph, round: int) -> Instruction:
    parameter = Parameter(f'theta_{round}')
    constraint_circuit = get_constraint_circuit(n, K, T, graph, state_prep_circuit=None, parameter=5*parameter)
    objective_circuit = get_objective_circuit(n, K, T, graph, state_prep_circuit=None, parameter=parameter)
    phase_operator = constraint_circuit.compose(
        objective_circuit, 
        qubits=range(objective_circuit.num_qubits)  
    )
    return phase_operator.to_instruction(label='phase_operator')
    

def get_mixer_instruction(n: int, T: int, round: int) -> Instruction:
    mixer_operator = get_mixer_operator(n, T, parameter=Parameter(f'beta_{round}'))
    return mixer_operator.to_instruction(label='mixer_operator')


def get_prog_qaoa_circuit(
    p: int,
    n: int,
    K: int,
    T: int,
    graph: nx.Graph
) -> QuantumCircuit:
    ceil_log_n2 = int(np.ceil(np.log2(n+2)))
    num_qubits = (K + T) * ceil_log_n2 + 1
    logger.info(f'Num qubits: {num_qubits}')
    state_prep_instruction = state_prep(n, T).to_instruction(label='state_prep')

    total_circuit = QuantumCircuit(num_qubits, T * ceil_log_n2)
    total_circuit.compose(
        state_prep_instruction,
        list(range(state_prep_instruction.num_qubits)),
        inplace=True
    )

    for i in range(p):
        phase_operator_instruction = get_phase_operator_instruction(n,K,T,graph,i)
        mixer_operator_instruction = get_mixer_instruction(n,T,i)
        total_circuit.compose(
            phase_operator_instruction,
            list(range(phase_operator_instruction.num_qubits)),
            inplace=True
        )
        total_circuit.compose(
            mixer_operator_instruction,
            list(range(mixer_operator_instruction.num_qubits)),
            inplace=True
        )
    logger.info(f'Total circuit has {total_circuit.num_qubits} qubits')
    return total_circuit