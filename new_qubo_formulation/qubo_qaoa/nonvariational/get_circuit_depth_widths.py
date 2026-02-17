import pickle

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import QAOAAnsatz

from qiskit_ibm_runtime import QiskitRuntimeService

from qopt_best_practices.sat_mapping import SATMapper

from qubo_qaoa.utils.swap_strategy import QUBOSwapStrategy
from qubo_qaoa.utils.lr_qaoa import get_LR_qaoa_circuit, get_hardware_LR_qaoa_circuit

from qiskit_qaoa.utils.hamiltonian_utils import get_Q_and_hamiltonian
from qiskit_qaoa.utils.circuit_graph_utils import circuit_to_graph, graph_to_operator
from qiskit_qaoa.utils.logging import get_logger

logger = get_logger(__name__)

def circuit_to_info(circuit: QuantumCircuit):
    return {
        'depth': circuit.depth(lambda instr: len(instr.qubits) > 1),
        'width': hamiltonian.num_qubits,
        'operations': circuit.count_ops(),
        'circuit': circuit
    }
    
p = 1
delta_b = 0.63
delta_g = 0.16


service = QiskitRuntimeService(name='us_instance')
backend = service.backend(name='ibm_boston')

data = {}

for filename in [
    'test_N2_W2','test_N3_W4','test_N3_W5','test_N4_W5','test_N4_W6',
    'test_N5_W6','test_N7_W2','test_N7_W3','test_N7_W4','test_N7_W5',
    'test_N8_W2','test_N8_W3','test_N8_W4','test_N8_W5','test_N8_W6',
    'test_N9_W6','test_N10_W6','test_N14_W7'
]:
    data_file = f'/lustre/scratch127/qpg/jc59/new_qubo_formulation/oriented/qubo_data/qubo_data_{filename}.gfa.pkl'
    try:
        Q, hamiltonian, offset, ising_offset = get_Q_and_hamiltonian(data_file)
        num_qubits: int = hamiltonian.num_qubits

        logger.info(f'Compiling {filename} with line SWAP strategy')
        swap_strat = QUBOSwapStrategy.from_line(range(num_qubits))
        edge_colouring = {(i, i+1): i % 2 for i in range(num_qubits)}
        edge_colouring.update({(i+1, i): i % 2 for i in range(num_qubits)})

        qc = QAOAAnsatz(
            cost_operator=hamiltonian,
            reps = 1,
            flatten=True
        )
        graph = circuit_to_graph(qc, qc.parameters[1])

        remapped_g, sat_map, min_sat_layers = SATMapper(timeout=60).remap_graph_with_sat(
            graph=graph, swap_strategy=swap_strat, max_layers = int(num_qubits)
        )
        if remapped_g is None or sat_map is None:
            raise Exception('Failed to find initial layout')

        cost_op = graph_to_operator(remapped_g, swap_strat._num_vertices)

        phis = ParameterVector('ϕ', num_qubits)
        fixed_hardware_qc, circuit = get_hardware_LR_qaoa_circuit(
            p, delta_b, delta_g, num_qubits,
            cost_op, sat_map, backend, edge_colouring, swap_strat,
            None, phis=phis,
        )

        fixed_qc, circuit = get_LR_qaoa_circuit(
            p, delta_b, delta_g, num_qubits,
            hamiltonian, None, phis=phis, measure=True
        )
        
        logger.info(f"""
                    filename: {filename},
                    Abstract circuit: depth {fixed_qc.depth()}, counts: {sum(fixed_qc.count_ops().values())},
                    Hardware circuit: depth {fixed_hardware_qc.depth()}, counts: {sum(fixed_hardware_qc.count_ops().values())}
                    """
        )

        data[filename] = {
            'abstract': circuit_to_info(fixed_qc),
            'hardware': circuit_to_info(fixed_hardware_qc)
        }
        
    except Exception as e:
        logger.info(e)
        pass
    
    
with open(f'/lustre/scratch127/qpg/jc59/new_qubo_formulation/oriented/circuit_depth_width/depths.p{p}.pkl', 'wb') as f:
    pickle.dump(data, f)