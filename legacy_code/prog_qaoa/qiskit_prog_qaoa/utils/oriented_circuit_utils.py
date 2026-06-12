from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter, Instruction, ParameterExpression
import numpy as np
import networkx as nx
from qiskit_prog_qaoa.utils.logging import get_logger

logger = get_logger(__name__)

def is_equal_to_without_parity(n: int, j: int) -> QuantumCircuit:
    """
    Creates a circuit that checks if a register encodes a value in binary and flips a flag if so.
    n: number of gfa segments
    j: node to check
    """
    ceil_log_n2 = int(np.ceil(np.log2(n+2)))
    circ = QuantumCircuit(ceil_log_n2 + 1)
    circ.mcx(list(range(ceil_log_n2)), ceil_log_n2, ctrl_state=j)
    return circ


def is_equal_to_with_parity(n: int, j: int, b: int):
    """
    Creates a circuit that checks if a register encodes a value in binary and flips a flag if so.
    n: number of gfa segments
    j: node to check
    b: orientation of node to check
    """
    if b not in (0, 1):
        raise Exception(f'b should be binary. Current: {b}')
    ceil_log_n2 = int(np.ceil(np.log2(n+2)))
    ctrl_state = 2 * j + b
    circ = QuantumCircuit(ceil_log_n2 + 1 + 1)
    circ.mcx(list(range(circ.num_qubits-1)), circ.num_qubits - 1, ctrl_state=ctrl_state)
    return circ


def controlled_copy_with_swap(circuit: QuantumCircuit, registers: dict, K: int, t: int, parameter: Parameter | ParameterExpression | None = None) -> QuantumCircuit:
    """
    Appends a controlled copy circuit that also shuffles all the registers in the copy list forward one place.
    |0>|to_be_copied>|copy_1>|copy_2>...|copy_K> -> |0>|to_be_copied>|copy_1>|copy_2>...|copy_K>
    |1>|to_be_copied>|copy_1>|copy_2>...|copy_K> -> |1>|to_be_copied>|copy_K+ to_be_copied>|copy_1 >...|copy_K-1> , where the addition is bitwise.
    
    If copy_K is not empty (i.e. all-zero ket), applies a penalty phase.
    n: number of gfa segments
    K: number of copy registers
    """
    # Penalise only when computing, not uncomputing, and only when trying to make a new copy.
    # If final register not empty, visited a node too many times. Apply a large penalty, since we cannot account for the next graph step. 
    # It also messes up already-tracked graph steps, overwriting them with a bitwise-and with the new step.
    # So apply 3x penalty.
    if parameter is not None:
        circuit.mcx(
            [circuit.find_bit(registers['flag'][0]).index] + list(range(circuit.find_bit(registers[f'next_node_{K-1}'][1]).index, circuit.find_bit(registers[f'next_node_{K-1}'][-1]).index + 1)), 
            circuit.find_bit(registers['visits_flag'][0]).index, 
            ctrl_state=1
        )
        circuit.cp(-parameter, circuit.find_bit(registers['flag'][0]).index, circuit.find_bit(registers['visits_flag'][0]).index)
        circuit.mcx(
            [circuit.find_bit(registers['flag'][0]).index] + list(range(circuit.find_bit(registers[f'next_node_{K-1}'][1]).index, circuit.find_bit(registers[f'next_node_{K-1}'][-1]).index + 1)), 
            circuit.find_bit(registers['visits_flag'][0]).index, 
            ctrl_state=1
        )
        circuit.x(circuit.find_bit(registers['visits_flag'][0]).index)
        circuit.cp(parameter, circuit.find_bit(registers['flag'][0]).index, circuit.find_bit(registers['visits_flag'][0]).index)
        circuit.x(circuit.find_bit(registers['visits_flag'][0]).index)
    
    # For each qubit i in final copy register:
    for i in range(registers[f'next_node_{K-1}'].size):
        # Swap down to ith position in the first copy register, conditioned on the c_copy_flag
        start_idx = circuit.find_bit(registers[f'next_node_{K-1}'][i]).index
        end_idx = circuit.find_bit(registers[f'next_node_{0}'][i]).index
        for idx in range(start_idx, end_idx, -1):
            circuit.cswap(circuit.find_bit(registers['flag'][0]).index, idx, idx-1)
            
    # Copy the to_be_copied register into the first next_node register, conditioned on the c_copy_flag
    for i in range(registers[f'solution_{t}'].size):
        circuit.ccx(
            circuit.find_bit(registers['flag'][0]).index, 
            circuit.find_bit(registers[f'solution_{t}'][i]).index, 
            circuit.find_bit(registers['next_node_0'][i]).index
        )
    return circuit


def compute_next_nodes_with_parity(
        circuit: QuantumCircuit, registers: dict, j: int, b:int, n: int, K: int, T: int, parameter: Parameter | ParameterExpression | None
) -> QuantumCircuit:
    """
    Appends a compute_next_nodes subroutine to a circuit, which initialises registers .
    For t in 0..T-2:
        Checks if the t th solution register encodes j and flips a flag if so.
        If the flag is set, shuffles the next_node registers forward one place and copies the (t+1)th solution register to the next_node registers.
        Resets the flag.
    """
    if b not in (0, 1):
        raise Exception(f'b should be binary. Current: {b}')
    
    is_equal_circ = is_equal_to_with_parity(n, j, b)
    for t in range(T-1):
        circuit.append(
            is_equal_circ, 
            list(range(circuit.find_bit(registers[f'solution_{t}'][0]).index, circuit.find_bit(registers[f'solution_{t}'][-1]).index + 1)) \
                + [circuit.find_bit(registers['flag'][0]).index],
        )

        circuit = controlled_copy_with_swap(circuit, registers, K, t+1, parameter)
        
        circuit.append(
            is_equal_circ, 
            list(range(circuit.find_bit(registers[f'solution_{t}'][0]).index, circuit.find_bit(registers[f'solution_{t}'][-1]).index + 1)) \
                + [circuit.find_bit(registers['flag'][0]).index]
        )
    return circuit


def uncompute_next_nodes_with_parity(
        circuit: QuantumCircuit, registers: dict, j: int, b:int, n: int, K: int, T: int
) -> QuantumCircuit:
    """
    Appends a compute_next_nodes subroutine to a circuit, which initialises registers .
    For t in 0..T-2:
        Checks if the t th solution register encodes j and flips a flag if so.
        If the flag is set, shuffles the next_node registers forward one place and copies the (t+1)th solution register to the next_node registers.
        Resets the flag.
    """
    if b not in (0, 1):
        raise Exception(f'b should be binary. Current: {b}')
    
    is_equal_circ = is_equal_to_with_parity(n, j, b)
    for t in range(T-2, -1, -1):
        circuit.append(
            is_equal_circ, 
            list(range(circuit.find_bit(registers[f'solution_{t}'][0]).index, circuit.find_bit(registers[f'solution_{t}'][-1]).index + 1)) \
                + [circuit.find_bit(registers['flag'][0]).index],
        )

        new_circuit = QuantumCircuit()
        for register in registers.values():
            new_circuit.add_register(register)
        new_circuit = controlled_copy_with_swap(new_circuit, registers, K, t+1, parameter=None)
        
        circuit.append(
            new_circuit.inverse(),
            new_circuit.qubits
        )
        
        circuit.append(
            is_equal_circ, 
            list(range(circuit.find_bit(registers[f'solution_{t}'][0]).index, circuit.find_bit(registers[f'solution_{t}'][-1]).index + 1)) \
                + [circuit.find_bit(registers['flag'][0]).index]
        )
    return circuit


def penalise_graph_steps(
        circuit: QuantumCircuit, registers: dict, j: int, b: int, parameter: Parameter | ParameterExpression, graph: nx.Graph, n: int, K:int
) -> QuantumCircuit:
    """
    Appends a penalise_graph_steps subroutine to a circuit, which penalises any step from node j to a node not adjacent to j.
    For each node i in 1..n not adjacent to j:
        For each possible number of visits k:
            Checks if the kth next node register is equal to j, and flips a flag if so.
            Applies a phase to the flag qubit proportional to the parameter.
            Resets the flag.
    """
    nodes = list(graph.nodes)
    for i in range(1, n+1):
        for bb in range(2):
            if (nodes[2*(j-1)+b], nodes[2*(i-1)+bb]) not in graph.edges:
                is_equal_circ = is_equal_to_with_parity(n, i, bb)
                for k in range(K):
                    circuit.append(
                        is_equal_circ,
                        list(range(
                            circuit.find_bit(registers[f'next_node_{k}'][0]).index, circuit.find_bit(registers[f'next_node_{k}'][-1]).index + 1
                        )) + [circuit.find_bit(registers['flag'][0]).index]
                    )
                    circuit.p(
                        parameter, 
                        circuit.find_bit(registers['flag'][0]).index
                    )
                    circuit.append(
                        is_equal_circ,
                        list(range(
                            circuit.find_bit(registers[f'next_node_{k}'][0]).index, circuit.find_bit(registers[f'next_node_{k}'][-1]).index + 1
                        )) + [circuit.find_bit(registers['flag'][0]).index]
                    )
    return circuit


def penalise_graph_end_steps(
    circuit: QuantumCircuit, registers: dict, parameter: Parameter | ParameterExpression, n: int, K: int
) -> QuantumCircuit:
    """
    Appends a penalise_graph_end_steps subroutine to a circuit, which penalises any step from the end node to a non-end node.
    For each node j in 1..n:
        For each possible number of visits k:
            Checks if the kth next node register is equal to j, and flips a flag if so.
            Applies a phase to the flag qubit proportional to the parameter.
            Resets the flag.
    """
    for i in range(1, n+1):
        is_equal_circ = is_equal_to_without_parity(n, i)
        for k in range(K):
            circuit.append(
                is_equal_circ,
                list(range(
                    circuit.find_bit(registers[f'next_node_{k}'][1]).index, circuit.find_bit(registers[f'next_node_{k}'][-1]).index + 1
                )) + [circuit.find_bit(registers['flag'][0]).index]
            )
            circuit.p(
                parameter, 
                circuit.find_bit(registers['flag'][0]).index
            )
            circuit.append(
                is_equal_circ,
                list(range(
                    circuit.find_bit(registers[f'next_node_{k}'][1]).index, circuit.find_bit(registers[f'next_node_{k}'][-1]).index + 1
                )) + [circuit.find_bit(registers['flag'][0]).index]
            )
    return circuit


def compute_next_nodes_without_parity(
        circuit: QuantumCircuit, registers: dict, j: int, n: int, K: int, T: int, parameter: Parameter | None
) -> QuantumCircuit:
    """
    Appends a compute_next_nodes subroutine to a circuit, which initialises registers .
    For t in 0..T-2:
        Checks if the t th solution register encodes j and flips a flag if so.
        If the flag is set, shuffles the next_node registers forward one place and copies the (t+1)th solution register to the next_node registers.
        Resets the flag.
    """    
    is_equal_circ = is_equal_to_without_parity(n, j)
    for t in range(T-1):
        circuit.append(
            is_equal_circ, 
            list(range(circuit.find_bit(registers[f'solution_{t}'][1]).index, circuit.find_bit(registers[f'solution_{t}'][-1]).index + 1)) \
                + [circuit.find_bit(registers['flag'][0]).index],
        )
        
        circuit = controlled_copy_with_swap(circuit, registers, K, t+1, parameter)
        
        circuit.append(
            is_equal_circ, 
            list(range(circuit.find_bit(registers[f'solution_{t}'][1]).index, circuit.find_bit(registers[f'solution_{t}'][-1]).index + 1)) \
                + [circuit.find_bit(registers['flag'][0]).index]
        )
    return circuit


def uncompute_next_nodes_without_parity(
        circuit: QuantumCircuit, registers: dict, j: int, n: int, K: int, T: int
) -> QuantumCircuit:
    """
    Appends an uncompute_next_nodes subroutine to a circuit, which resets registers .
    For t in 0..T-2:
        Checks if the t th solution register encodes j and flips a flag if so.
        If the flag is set, shuffles the next_node registers forward one place and copies the (t+1)th solution register to the next_node registers.
        Resets the flag.
    """
    
    is_equal_circ = is_equal_to_without_parity(n, j)
    for t in range(T-2,-1,-1):
        circuit.append(
            is_equal_circ, 
            list(range(circuit.find_bit(registers[f'solution_{t}'][1]).index, circuit.find_bit(registers[f'solution_{t}'][-1]).index + 1)) \
                + [circuit.find_bit(registers['flag'][0]).index],
        )

        new_circuit = QuantumCircuit()
        for register in registers.values():
            new_circuit.add_register(register)
        new_circuit = controlled_copy_with_swap(new_circuit, registers, K, t+1, parameter=None)
        
        circuit.append(
            new_circuit.inverse(),
            new_circuit.qubits
        )
        
        circuit.append(
            is_equal_circ, 
            list(range(circuit.find_bit(registers[f'solution_{t}'][1]).index, circuit.find_bit(registers[f'solution_{t}'][-1]).index + 1)) \
                + [circuit.find_bit(registers['flag'][0]).index]
        )
    return circuit


def get_constraint_circuit(
        n: int,
        K: int,
        T: int,
        graph: nx.Graph,
        parameter: Parameter | ParameterExpression =Parameter('theta_cons'),
        state_prep_circuit: QuantumCircuit | None = None,
) -> QuantumCircuit:
    """
    Prepares a quantum circuit for the constraint function for Oriented Tangle Prog-QAOA.
    """
    ceil_log_n2 = int(np.ceil(np.log2(n+2)))
    circuit = QuantumCircuit()

    registers = {f'solution_{t}' : QuantumRegister(ceil_log_n2+1, name=f'solution_{t}') for t in range(T)}
    registers.update({f'next_node_{k}': QuantumRegister(ceil_log_n2+1, name=f'next_node_{k}') for k in range(K)})
    registers.update({'flag': QuantumRegister(1, name='flag')})
    registers.update({'visits_flag': QuantumRegister(1, name='visits_flag')})


    for register in registers.values():
        circuit.add_register(register)

    if state_prep_circuit is not None:
        circuit.append(state_prep_circuit, list(range(T * ceil_log_n2)))

    for j in range(1, n+1):
        for b in range(2):
            circuit = compute_next_nodes_with_parity(circuit, registers, j, b, n, K, T, 3*parameter)
            circuit = penalise_graph_steps(circuit, registers, j, b, parameter, graph, n, K)
            circuit = uncompute_next_nodes_with_parity(circuit, registers, j, b, n, K, T)

    # Walk is allowed to stay in end indefinitely. end or end+end are not penalised, so any "allowed" steps are never penalised.
    # Could miss a case where the path leaves end node several times, each occuring at the
    # same point mod K, and the next nodes bitwise-sum to 0 or end.
    circuit = compute_next_nodes_without_parity(circuit, registers, n+1, n, K, T, parameter=None)
    circuit = penalise_graph_end_steps(circuit, registers, parameter, n, K)
    circuit = uncompute_next_nodes_without_parity(circuit, registers, n+1, n, K, T)
    return circuit


def compute_count(circuit: QuantumCircuit, registers: dict, j: int, n: int, T: int) -> QuantumCircuit:
    """
    Appends a compute_count subroutine to a circuit.
    For each register x_0 ... x_{T-1}, checks if the register encodes j in binary (ignoring the orientation) and flips a flag if so.
    Adds 1 to a count register based on the flag.
    Resets the flag.
    """
    is_equal_circ = is_equal_to_without_parity(n, j)

    for t in range(T):
        circuit.append(
            is_equal_circ, 
            list(range(circuit.find_bit(registers[f'solution_{t}'][1]).index, circuit.find_bit(registers[f'solution_{t}'][-1]).index + 1)) \
                + [circuit.find_bit(registers['flag'][0]).index]
        )
        
        for i in range(registers['count'].size-1,0,-1):
            circuit.mcx(
                [circuit.find_bit(registers['flag'][0]).index] + list(range(circuit.find_bit(registers['count'][0]).index, circuit.find_bit(registers['count'][i]).index)),
                circuit.find_bit(registers['count'][i]).index
            )
        circuit.cx(circuit.find_bit(registers['flag'][0]).index, circuit.find_bit(registers['count'][0]).index)
        circuit.append(
            is_equal_circ, 
            list(range(circuit.find_bit(registers[f'solution_{t}'][1]).index, circuit.find_bit(registers[f'solution_{t}'][-1]).index + 1)) \
                + [circuit.find_bit(registers['flag'][0]).index],
            
        )
        
    return circuit


def uncompute_count(circuit: QuantumCircuit, registers: dict, j: int, n: int, T: int) -> QuantumCircuit:
    """
    Appends an uncompute_count subroutine to a circuit.
    For each register x_0 ... x_{T-1}, checks if the register encodes j in binary and flips a flag if so.
    Subtracts 1 to a count register based on the flag.
    Resets the flag.
    """
    is_equal_circ = is_equal_to_without_parity(n, j)

    for t in range(T):
        circuit.append(
            is_equal_circ, 
            list(range(circuit.find_bit(registers[f'solution_{t}'][1]).index, circuit.find_bit(registers[f'solution_{t}'][-1]).index + 1)) \
                + [circuit.find_bit(registers['flag'][0]).index],
        )
        
        circuit.cx(circuit.find_bit(registers['flag'][0]).index, circuit.find_bit(registers['count'][0]).index)
        for i in range(1, registers['count'].size):
            circuit.mcx(
                [circuit.find_bit(registers['flag'][0]).index] + list(range(circuit.find_bit(registers['count'][0]).index, circuit.find_bit(registers['count'][i]).index)),
                circuit.find_bit(registers['count'][i]).index
            )

        circuit.append(
            is_equal_circ, 
            list(range(circuit.find_bit(registers[f'solution_{t}'][1]).index, circuit.find_bit(registers[f'solution_{t}'][-1]).index + 1)) \
                + [circuit.find_bit(registers['flag'][0]).index]
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
    for i in range(2 ** ceil_log_K1):
        if not graph.nodes[nodes[2*(j-1)]]["weight"] - i == 0:
            circuit.mcx(
                list(range(circuit.find_bit(registers['count'][0]).index, circuit.find_bit(registers['count'][-1]).index + 1)),
                circuit.find_bit(registers['flag'][0]).index,
                ctrl_state=i
            )
            circuit.p(parameter * (graph.nodes[nodes[2*(j-1)]]["weight"] - i) ** 2, circuit.find_bit(registers['flag'][0]).index)
            circuit.mcx(
                list(range(circuit.find_bit(registers['count'][0]).index, circuit.find_bit(registers['count'][-1]).index + 1)),
                circuit.find_bit(registers['flag'][0]).index,
                ctrl_state=i
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
    Prepares a quantum circuit for the objective function for Oriented Tangle Prog-QAOA.
    """
    circuit = QuantumCircuit()

    ceil_log_n2 = int(np.ceil(np.log2(n+2)))
    ceil_log_K1 = int(np.ceil(np.log2(K+1)))
    registers = {f'solution_{t}' : QuantumRegister(ceil_log_n2+1, name=f'solution_{t}') for t in range(T)}
    registers.update({'flag': QuantumRegister(1, name='flag')})
    registers.update({'count': QuantumRegister(ceil_log_K1, name='count')})

    for register in registers.values():
        circuit.add_register(register)

    if state_prep_circuit is not None:
        circuit.append(state_prep_circuit, list(range(state_prep_circuit.num_qubits)))

    for j in range(1, n+1):
        circuit = compute_count(circuit, registers, j, n, T)
        circuit = penalise_count(circuit, registers, j, parameter, graph, K)
        circuit = uncompute_count(circuit, registers, j, n, T)


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
                
            for i in range(num_qubits-1,0,-1):
                circuit.mcx(list(range(i)), i)
            circuit.x(0)
            
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

    for i in range(num_qubits-1,0,-1):
        circuit.mcx(list(range(i)), i)
    circuit.x(0)
    return circuit


def state_prep(n: int, T: int) -> QuantumCircuit:
    ceil_log_n2 = int(np.ceil(np.log2(n+2)))
    uni = uniform_over_range(ceil_log_n2, n+1)
    circuit = QuantumCircuit((ceil_log_n2+1) * T)
    for t in range(T):
        circuit.h(t*(ceil_log_n2+1))
        circuit.append(
            uni,
            list(range(t *(ceil_log_n2+1) + 1, (t+1)*(ceil_log_n2+1)))   
        )
    return circuit


def get_mixer_operator(n: int, T: int, parameter=Parameter('beta')) -> QuantumCircuit:
    num_qubits = int(np.ceil(np.log2(n+2)) + 1) * T
    state_prep_circuit = state_prep(n, T)
    mixer = QuantumCircuit(num_qubits)
    mixer.append(
        state_prep_circuit.inverse(),
        range(num_qubits)
    )
    mixer.x(-1)
    mixer.mcp(-parameter, list(range(num_qubits - 1)), -1, ctrl_state=0)
    mixer.x(-1)
    mixer.append(
        state_prep_circuit,
        range(num_qubits)
    )
    return mixer


def get_phase_operator_gate(n: int, K: int, T: int, graph: nx.Graph, lamda: float, round: int) -> Instruction:
    parameter = Parameter(f'theta_{round}')
    constraint_circuit = get_constraint_circuit(n, K, T, graph, state_prep_circuit=None, parameter=lamda*parameter)
    objective_circuit = get_objective_circuit(n, K, T, graph, state_prep_circuit=None, parameter=parameter)
    circuit = QuantumCircuit(max(constraint_circuit.num_qubits, objective_circuit.num_qubits))
    
    circuit.append(constraint_circuit, list(range(constraint_circuit.num_qubits)))
    circuit.append(objective_circuit, list(range(objective_circuit.num_qubits)))
    
    return circuit.to_instruction(label='phase_operator')
    

def get_mixer_gate(n: int, T: int, round: int) -> Instruction:
    mixer_operator = get_mixer_operator(n, T, parameter=Parameter(f'beta_{round}'))
    return mixer_operator.to_instruction(label='mixer_operator')


def get_prog_qaoa_circuit(
    p: int,
    n: int,
    K: int,
    T: int,
    graph: nx.Graph,
    lamda: float
) -> QuantumCircuit:
    ceil_log_n2 = int(np.ceil(np.log2(n+2)))
    num_qubits = (K + T) * (ceil_log_n2 +1) + 1 + 1
    logger.info(f'Num qubits: {num_qubits}')
    state_prep_gate = state_prep(n, T).to_instruction(label='state_prep')

    total_circuit = QuantumCircuit(num_qubits)
    total_circuit.append(
        state_prep_gate,
        list(range(state_prep_gate.num_qubits))
    )

    for i in range(p):
        phase_operator_instruction = get_phase_operator_gate(n, K, T, graph, lamda, i)
        mixer_operator_instruction = get_mixer_gate(n, T, i)
        if num_qubits > max(phase_operator_instruction.num_qubits, mixer_operator_instruction.num_qubits):
            logger.error('Total circuit has spare qubits')
        total_circuit.append(
            phase_operator_instruction,
            list(range(phase_operator_instruction.num_qubits))
        )
        total_circuit.append(
            mixer_operator_instruction,
            list(range(mixer_operator_instruction.num_qubits))
        )
    return total_circuit