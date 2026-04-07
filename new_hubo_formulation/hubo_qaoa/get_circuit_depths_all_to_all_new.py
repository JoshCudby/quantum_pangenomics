import pickle

from sys import maxsize

from typing import TypedDict

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.circuit.library import CXGate, PauliEvolutionGate, QAOAAnsatz

from qiskit_aer import AerSimulator
from qiskit_aer.backends.backendconfiguration import AerBackendConfiguration

from qiskit.transpiler import PassManager, Layout
from qiskit.transpiler.passes import InverseCancellation, CommutativeCancellation
from qopt_best_practices.transpilation.swap_cancellation_pass import SwapToFinalMapping

from qiskit_qaoa.utils.transpiler_passes import ExtendedSwapStrategy, FindCommutingPauliEvolutionsMulti
from qiskit_qaoa.utils.commuting_gate_router_new import CommutingGateRouterNew
from qiskit_qaoa.utils.hamiltonian_utils import hamiltonian_to_interactions

from hubo_qaoa.utils.graph_to_hubo_hamiltonian import graph_to_hubo_hamiltonian
from hubo_qaoa.utils.gfa_utils import gfa_file_to_graph
from qiskit_qaoa.utils.logging import get_logger

logger = get_logger(__name__)


    
Best = TypedDict('Best', {'layout': Layout, 'depth': int, 'count': int, 'layers': int, 'circuit': QuantumCircuit})    


def sweep_swap_depths(layers: list[int], best_rz: Best):
    for layer in layers:
        pm = PassManager(
            [
                FindCommutingPauliEvolutionsMulti(), 
                CommutingGateRouterNew(
                    ess,
                    max_layers=layer,
                    perform_extra_swaps=True
                ),
                SwapToFinalMapping(),
                InverseCancellation(gates_to_cancel=[CXGate()]),
                CommutativeCancellation(basis_gates=["cx", "swap", "rz", "rzz"]),
                InverseCancellation(gates_to_cancel=[CXGate()]),
            ]
        )
        
        logger.info('Using trivial layout')
        layout = Layout({donor_qc.qubits[i]: i for i in range(num_physical_qubits)})
        
        qc = QuantumCircuit(num_physical_qubits)
        qc.append(PauliEvolutionGate(hamiltonian, time=Parameter('γ')), [layout.get_virtual_bits()[donor_qc.qubits[i]] for i in range(num_physical_qubits)])

        logger.info('Compiling with new Rz')
        tqc = pm.run(qc)   
        
        rz_depth = tqc.depth(lambda instr: len(instr.qubits) > 1)
        if rz_depth < best_rz['depth']:
            best_rz['depth'] = rz_depth
            best_rz['count'] = tqc.num_nonlocal_gates()
            best_rz['layers'] = layer
            best_rz['layout'] = layout 
            best_rz['circuit'] = tqc
            
    return


method = 'statevector'
backend_options = dict(
    method=method,
    device='GPU',
    precision='single',
    basis_gates=["sx", "x", "rz", "rzz", "cz", "id", "cx"]
)
results = {}

for filename, copy_numbers in zip(
    [
        # 'test_N2_W2', 'trivial', 
        # 'test_N3_W4', 
        # 'test_N4_W5', 
        # 'test_N4_W6', 'test_N5_W6', 
        # 'test_N7_W2', 'test_N7_W3','test_N7_W4', 
        # 'test_N7_W5', 
        # 'test_N8_W2', 'test_N8_W3','test_N8_W4', 
        # 'test_N8_W5', 
        # 'test_N8_W6',
        'test_N9_W6', 'test_N10_W6','test_N14_W7'
    ], 
    [
        # [1,1], [1,1,1], 
        # [2,1,1], 
        # [2,1,1,1],
        # [2,2,1,1], [1,2,1,1,1], 
        # [1,0,0,0,0,0,1], [1,1,0,0,0,0,1], [1,1,1,0,0,0,1], 
        # [1,1,1,0,1,0,1],
        # [1,0,0,0,0,0,0,1],[1,1,0,0,0,0,0,1],[1,1,1,0,0,0,0,1],
        # [1,1,1,1,0,0,0,1],
        # [1,1,0,1,1,1,0,1],
        [1,1,0,0,1,0,1,1,1], [1,1,0,0,1,0,1,1,0,1], [1,1,0,0,1,0,1,0,0,1,0,0,1,1]
    ]
):
    logger.info('-------------------------------------')
    logger.info(filename)
    logger.info('\n\n')
    filepath = f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/data/{filename}.gfa'
    graph, n, V, T = gfa_file_to_graph(filepath, copy_numbers)
    hamiltonian, norm = graph_to_hubo_hamiltonian(graph, n, T, lamda=10, constraint_terms=1.0)
    hamiltonian = hamiltonian * norm
    num_virtual_qubits: int = hamiltonian.num_qubits
    
    logger.info('Using all2all')
    ess = ExtendedSwapStrategy.from_all_to_all(n*T)

        
    num_physical_qubits = ess._num_vertices
    donor_qc = QuantumCircuit(num_physical_qubits)

    program_interactions = hamiltonian_to_interactions(hamiltonian, 0, 1.0)

    config = AerSimulator._DEFAULT_CONFIGURATION
    config["n_qubits"] = num_physical_qubits
    config = AerBackendConfiguration.from_dict(config)
    backend = AerSimulator(configuration=config, coupling_map=ess._coupling_map, **backend_options)
    backend.set_option("n_qubits", num_physical_qubits)

    default_qaoa = QAOAAnsatz(hamiltonian, reps=1, initial_state= QuantumCircuit(num_physical_qubits), mixer_operator=QuantumCircuit(num_physical_qubits))
    t_default_qaoa = transpile(default_qaoa, backend=backend, optimization_level=3, basis_gates=["sx", "x", "rz", "rzz", "cz", "id", "cx"])


    best_rz = Best(count=maxsize, depth=maxsize, layers=0, layout=Layout({donor_qc.qubits[i]: i for i in range(num_physical_qubits)}),circuit=QuantumCircuit(num_physical_qubits))
    layers = [0]

    sweep_swap_depths(layers, best_rz)       
    
    
    results[filename] = {
        'default': Best(
            count=t_default_qaoa.num_nonlocal_gates(), 
            depth=t_default_qaoa.depth(lambda instr: len(instr.qubits) > 1), 
            layers=0, 
            layout=Layout({donor_qc.qubits[i]: i for i in range(num_physical_qubits)}), 
            circuit=t_default_qaoa
        ),
        'rz': best_rz,
    }
    logger.info(f'Default: {t_default_qaoa.num_nonlocal_gates()}, custom: {best_rz["count"]}')
    
    try:
        with open('/lustre/scratch127/qpg/jc59/new_hubo_formulation/circuit_depths/results.all2all.new.pkl', 'rb') as f:
            loaded_results = pickle.load(f)
    except FileNotFoundError:
        loaded_results = dict()
    to_save = dict(loaded_results, **results)
        

    with open('/lustre/scratch127/qpg/jc59/new_hubo_formulation/circuit_depths/results.all2all.new.pkl', 'wb') as f:
        pickle.dump(to_save, f)
    
