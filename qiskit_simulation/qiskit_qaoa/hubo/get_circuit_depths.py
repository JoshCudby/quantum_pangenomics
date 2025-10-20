import numpy as np
import pickle

import argparse

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import CXGate, PauliEvolutionGate, QAOAAnsatz

from qiskit_aer import AerSimulator
from qiskit_aer.backends.backendconfiguration import AerBackendConfiguration

from qiskit.transpiler import PassManager, Layout
from qiskit.transpiler.passes import InverseCancellation, CommutativeCancellation
from qopt_best_practices.transpilation.swap_cancellation_pass import SwapToFinalMapping

from qiskit_qaoa.utils.transpiler_passes import ExtendedSwapStrategy, FindCommutingPauliEvolutionsMulti
from qiskit_qaoa.utils.commuting_gate_router_rzz import CommutingGateRouterRzz
from qiskit_qaoa.utils.commuting_gate_router import CommutingGateRouter
from qiskit_qaoa.hubo.graph_to_hubo_hamiltonian import graph_to_hubo_hamiltonian
from qiskit_qaoa.utils.gfa_utils import gfa_file_to_graph
from qiskit_qaoa.utils.sat_mapper import HigherOrderSatMapper
from qiskit_qaoa.utils.hamiltonian_utils import hamiltonian_to_interactions
from qiskit_qaoa.utils.logging import get_logger

logger = get_logger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--timeout', type=int)
args = parser.parse_args()

method = 'statevector'
backend_options = dict(
    method=method,
    device='GPU',
    precision='single',
    basis_gates=["sx", "x", "rz", "rzz", "cz", "id", "cx"]
)
results = {}
mapper = HigherOrderSatMapper(timeout=args.timeout)

for filename, copy_numbers in zip(
    [
        'test_N2_W2', 'trivial', 'test_N3_W4', 'test_N4_W5', 
        'test_N4_W6', 
        'test_N5_W6', 'test_N7_W2', 'test_N7_W3','test_N7_W4', 
        'test_N7_W5', 
        'test_N8_W2', 'test_N8_W3','test_N8_W4', 
        'test_N8_W5', 
        'test_N8_W6',
        # 'test_N9_W6', 'test_N10_W6','test_N14_W7'
    ], 
    [
        [1,1], [1,1,1], [2,1,1], [2,1,1,1],
        [2,2,1,1],
        [1,2,1,1,1], [1,0,0,0,0,0,1], [1,1,0,0,0,0,1], [1,1,1,0,0,0,1], 
        [1,1,1,0,1,0,1],
        [1,0,0,0,0,0,0,1],[1,1,0,0,0,0,0,1],[1,1,1,0,0,0,0,1],
        [1,1,1,1,0,0,0,1],
        [1,1,0,1,1,1,0,1],
        # [1,1,0,0,1,0,1,1,1], [1,1,0,0,1,0,1,1,0,1], [1,1,0,0,1,0,1,0,0,1,0,0,1,1]
    ]
):
    logger.info('-------------------------------------')
    logger.info(filename)
    logger.info('\n\n')
    filepath = f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/data/{filename}.gfa'
    graph, n, V, T = gfa_file_to_graph(filepath, copy_numbers)
    hamiltonian = graph_to_hubo_hamiltonian(graph, n, T, lamda=10, constraint_terms=1.0)
    ess = ExtendedSwapStrategy.from_grid(n, T)
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


    best_rzz_depth = np.inf
    best_rzz_count = np.inf
    best_rzz_layers = 0
    best_rz_depth = np.inf
    best_rz_count = np.inf
    best_rz_layers = 0

    layers = sorted(list(set([int(x) for x in np.linspace(0, len(ess._swap_layers), 10)])))
    for layer in layers:
        pm = PassManager(
            [
                FindCommutingPauliEvolutionsMulti(), 
                CommutingGateRouterRzz(
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
        pm_rz = PassManager(
            [
                FindCommutingPauliEvolutionsMulti(), 
                CommutingGateRouter(
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

        if args.timeout == 0:
            logger.info('Using trivial layout')
            layout = Layout({donor_qc.qubits[i]: i for i in range(num_physical_qubits)})
        else:
            sat_results = mapper.hubo_max_sat(
                num_physical_qubits, program_interactions, ess, layer
            )
            if sat_results is None:
                logger.info('No results')
                continue

            mapping = sat_results[layer][1]
            edge_map = dict(mapping)
            
            layout = Layout({donor_qc.qubits[key]: val for key, val in edge_map.items()})

        qc = QuantumCircuit(num_physical_qubits)
        qc.append(PauliEvolutionGate(hamiltonian), [layout.get_virtual_bits()[donor_qc.qubits[i]] for i in range(num_physical_qubits)])

        logger.info('Compiling with Rzz')
        tqc = pm.run(qc)
        logger.info('Compiling with Rz')
        tqc_rz = pm_rz.run(qc)

        rzz_depth = tqc.depth(lambda instr: len(instr.qubits) > 1)
        if rzz_depth < best_rzz_depth:
            best_rzz_depth = rzz_depth
            best_rzz_count = tqc.num_nonlocal_gates()
            best_rzz_layers = layer

        rz_depth = tqc_rz.depth(lambda instr: len(instr.qubits) > 1)
        if rz_depth < best_rz_depth:
            best_rz_depth = rz_depth
            best_rz_count = tqc_rz.num_nonlocal_gates()
            best_rz_layers = layer


    best_rzz_index = layers.index(best_rzz_layers)
    rzz_fine_layers = sorted(list(set([int(x) for x in np.linspace(layers[max(best_rzz_index - 1, 0)]+1, layers[min(best_rzz_index + 1, len(layers)-1)]-1, 10)])))
    logger.info(f'Fine Rzz search. Best rzz layers: {best_rzz_layers}. Searching: {rzz_fine_layers}')
    for layer in rzz_fine_layers:
        pm = PassManager(
            [
                FindCommutingPauliEvolutionsMulti(), 
                CommutingGateRouterRzz(
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

        if args.timeout == 0:
            layout = Layout({donor_qc.qubits[i]: i for i in range(num_physical_qubits)})
        else:
            sat_results = mapper.hubo_max_sat(
                num_physical_qubits, program_interactions, ess, layer
            )
            if sat_results is None:
                logger.info('No results')
                continue

            mapping = sat_results[layer][1]
            edge_map = dict(mapping)
            layout = Layout({donor_qc.qubits[key]: val for key, val in edge_map.items()})

        qc = QuantumCircuit(num_physical_qubits)
        qc.append(PauliEvolutionGate(hamiltonian), [layout.get_virtual_bits()[donor_qc.qubits[i]] for i in range(num_physical_qubits)])

        logger.info('Compiling with Rzz')
        tqc = pm.run(qc)

        rzz_depth = tqc.depth(lambda instr: len(instr.qubits) > 1)
        if rzz_depth < best_rzz_depth:
            best_rzz_depth = rzz_depth
            best_rzz_count = tqc.num_nonlocal_gates()
            best_rzz_layers = layer
    
    
    best_rz_index = layers.index(best_rz_layers)
    rz_fine_layers = sorted(list(set([int(x) for x in np.linspace(layers[best_rz_index - 1]+1, layers[best_rz_index + 1]-1, 10)])))
    logger.info(f'Fine Rz search. Best rz layers: {best_rz_layers}. Searching: {rz_fine_layers}')

    for layer in rz_fine_layers:
        pm_rz = PassManager(
            [
                FindCommutingPauliEvolutionsMulti(), 
                CommutingGateRouter(
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

        if args.timeout == 0:
            layout = Layout({donor_qc.qubits[i]: i for i in range(num_physical_qubits)})
        else:
            sat_results = mapper.hubo_max_sat(
                num_physical_qubits, program_interactions, ess, layer
            )
            if sat_results is None:
                logger.info('No results')
                continue

            mapping = sat_results[layer][1]
            edge_map = dict(mapping)
            layout = Layout({donor_qc.qubits[key]: val for key, val in edge_map.items()})

        qc = QuantumCircuit(num_physical_qubits)
        qc.append(PauliEvolutionGate(hamiltonian), [layout.get_virtual_bits()[donor_qc.qubits[i]] for i in range(num_physical_qubits)])

        logger.info('Compiling with Rz')
        tqc_rz = pm_rz.run(qc)


        rz_depth = tqc_rz.depth(lambda instr: len(instr.qubits) > 1)
        if rz_depth < best_rz_depth:
            best_rz_depth = rz_depth
            best_rz_count = tqc_rz.num_nonlocal_gates()
            best_rz_layers = layer
             
    
    results[filename] = {
        'default': (t_default_qaoa.num_nonlocal_gates(), t_default_qaoa.depth(lambda instr: len(instr.qubits) > 1)),
        'rz': (best_rz_count, best_rz_depth, best_rz_layers),
        'rzz': (best_rzz_count, best_rzz_depth, best_rzz_layers),
    }
    
    try:
        with open(f'/lustre/scratch127/qpg/jc59/circuit_depths/results{args.timeout}.pkl', 'rb') as f:
            loaded_results = pickle.load(f)
    except FileNotFoundError:
        loaded_results = dict()
    to_save = dict(loaded_results, **results)
        

    with open(f'/lustre/scratch127/qpg/jc59/circuit_depths/results{args.timeout}.pkl', 'wb') as f:
        pickle.dump(to_save, f)