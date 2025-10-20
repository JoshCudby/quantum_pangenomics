import numpy as np
import pickle

from sys import maxsize

import argparse

from typing import TypedDict

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import CXGate, PauliEvolutionGate, QAOAAnsatz

from qiskit_aer import AerSimulator
from qiskit_aer.backends.backendconfiguration import AerBackendConfiguration

from qiskit.transpiler import PassManager, Layout
from qiskit.transpiler.passes import InverseCancellation, CommutativeCancellation
from qopt_best_practices.transpilation.swap_cancellation_pass import SwapToFinalMapping

from qiskit_qaoa.utils.transpiler_passes import ExtendedSwapStrategy, FindCommutingPauliEvolutionsMulti
from qiskit_qaoa.utils.commuting_gate_router_precompute import CommutingGateRouterPrecompute
from qiskit_qaoa.utils.commuting_gate_router_precompute_rzz import CommutingGateRouterPrecomputeRzz

from qiskit_qaoa.hubo.graph_to_hubo_hamiltonian import graph_to_hubo_hamiltonian
from qiskit_qaoa.utils.gfa_utils import gfa_file_to_graph
from qiskit_qaoa.utils.sat_mapper import HigherOrderSatMapper
from qiskit_qaoa.utils.hamiltonian_utils import hamiltonian_to_interactions
from qiskit_qaoa.utils.logging import get_logger

logger = get_logger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('-C', '--coupling', type=str, default='grid')
parser.add_argument('-t', '--timeout', type=int)
args = parser.parse_args()

    
Best = TypedDict('Best', {'layout': Layout, 'depth': int, 'count': int, 'layers': int, 'circuit': QuantumCircuit})    


def sweep_swap_depths(layers: list[int], best_rz: Best, best_rzz: Best):
    for layer in layers:
        pm = PassManager(
            [
                FindCommutingPauliEvolutionsMulti(), 
                CommutingGateRouterPrecompute(
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
        pm_rzz = PassManager(
            [
                FindCommutingPauliEvolutionsMulti(), 
                CommutingGateRouterPrecomputeRzz(
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

        logger.info('Compiling with precompute Rz')
        tqc = pm.run(qc)   
        
        rz_depth = tqc.depth(lambda instr: len(instr.qubits) > 1)
        if rz_depth < best_rz['depth']:
            best_rz['depth'] = rz_depth
            best_rz['count'] = tqc.num_nonlocal_gates()
            best_rz['layers'] = layer
            best_rz['layout'] = layout 
            best_rz['circuit'] = tqc
            
        logger.info('Compiling with precompute Rzz')
        tqc_rzz = pm_rzz.run(qc)   
        
        rzz_depth = tqc_rzz.depth(lambda instr: len(instr.qubits) > 1)
        if rzz_depth < best_rzz['depth']:
            best_rzz['depth'] = rzz_depth
            best_rzz['count'] = tqc_rzz.num_nonlocal_gates()
            best_rzz['layers'] = layer
            best_rzz['layout'] = layout 
            best_rzz['circuit'] = tqc
    return


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
        'test_N2_W2', 'trivial', 
        'test_N3_W4', 
        'test_N4_W5', 'test_N4_W6', 
        # 'test_N5_W6', 'test_N7_W2', 'test_N7_W3','test_N7_W4', 
        # 'test_N7_W5', 
        # 'test_N8_W2', 'test_N8_W3','test_N8_W4', 
        # 'test_N8_W5', 
        # 'test_N8_W6',
        # 'test_N9_W6', 'test_N10_W6','test_N14_W7'
    ], 
    [
        [1,1], [1,1,1], 
        [2,1,1], 
        [2,1,1,1],[2,2,1,1],
        # [1,2,1,1,1], [1,0,0,0,0,0,1], [1,1,0,0,0,0,1], [1,1,1,0,0,0,1], 
        # [1,1,1,0,1,0,1],
        # [1,0,0,0,0,0,0,1],[1,1,0,0,0,0,0,1],[1,1,1,0,0,0,0,1],
        # [1,1,1,1,0,0,0,1],
        # [1,1,0,1,1,1,0,1],
        # [1,1,0,0,1,0,1,1,1], [1,1,0,0,1,0,1,1,0,1], [1,1,0,0,1,0,1,0,0,1,0,0,1,1]
    ]
):
    logger.info('-------------------------------------')
    logger.info(filename)
    logger.info('\n\n')
    filepath = f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/data/{filename}.gfa'
    graph, n, V, T = gfa_file_to_graph(filepath, copy_numbers)
    hamiltonian = graph_to_hubo_hamiltonian(graph, n, T, lamda=10, constraint_terms=1.0)
    
    if args.coupling == 'all2all':
        logger.info('Using all2all')
        ess = ExtendedSwapStrategy.from_all_to_all(n*T)
    else:
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


    best_rz = Best(count=maxsize, depth=maxsize, layers=0, layout=Layout({donor_qc.qubits[i]: i for i in range(num_physical_qubits)}),circuit=QuantumCircuit(num_physical_qubits))
    best_rzz = Best(count=maxsize, depth=maxsize, layers=0, layout=Layout({donor_qc.qubits[i]: i for i in range(num_physical_qubits)}),circuit=QuantumCircuit(num_physical_qubits))
    if args.coupling == 'all2all':
        layers = [0]
    else:
        layers = sorted(list(set([int(x) for x in np.linspace(0, len(ess._swap_layers), 10)])))
        
    sweep_swap_depths(layers, best_rz, best_rzz)
        
    
    if not args.coupling == 'all2all':
        best_rz_index = layers.index(best_rz['layers'])
        rz_fine_layers = sorted(list(set([int(x) for x in np.linspace(layers[max(best_rz_index - 1, 0)]+1, layers[min(best_rz_index + 1, len(layers)-1)]-1, 5)])))
        
        best_rzz_index = layers.index(best_rzz['layers'])
        rzz_fine_layers = sorted(list(set([int(x) for x in np.linspace(layers[max(best_rzz_index - 1, 0)]+1, layers[min(best_rzz_index + 1, len(layers)-1)]-1, 5)])))

        fine_layers = sorted(set(rz_fine_layers + rzz_fine_layers))
        logger.info(f'Fine search.  Best rz layers: {best_rz["layers"]}. Best rzz layers: {best_rzz["layers"]}. Searching: {fine_layers}')
        sweep_swap_depths(rzz_fine_layers, best_rz, best_rzz)             
    
    
    results[filename] = {
        'default': (t_default_qaoa.num_nonlocal_gates(), t_default_qaoa.depth(lambda instr: len(instr.qubits) > 1)),
        'rz': list(best_rz.values()),
        'rzz': list(best_rzz.values()),
    }
    
    if args.coupling == 'all2all':
        try:
            with open(f'/lustre/scratch127/qpg/jc59/circuit_depths/results.all2all.precompute.{args.timeout}.pkl', 'rb') as f:
                loaded_results = pickle.load(f)
        except FileNotFoundError:
            loaded_results = dict()
        to_save = dict(loaded_results, **results)
            

        with open(f'/lustre/scratch127/qpg/jc59/circuit_depths/results.all2all.precompute.{args.timeout}.pkl', 'wb') as f:
            pickle.dump(to_save, f)
    
    else:        
        try:
            with open(f'/lustre/scratch127/qpg/jc59/circuit_depths/results.precompute.{args.timeout}.pkl', 'rb') as f:
                loaded_results = pickle.load(f)
        except FileNotFoundError:
            loaded_results = dict()
        to_save = dict(loaded_results, **results)
            

        with open(f'/lustre/scratch127/qpg/jc59/circuit_depths/results.precompute.{args.timeout}.pkl', 'wb') as f:
            pickle.dump(to_save, f)