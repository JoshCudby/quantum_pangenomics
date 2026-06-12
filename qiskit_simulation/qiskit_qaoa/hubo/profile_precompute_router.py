"""Profile the CommutingGateRouterPrecompute transpiler pass on a small instance.

Minimal script for timing and debugging ``CommutingGateRouterPrecompute`` on
a hard-coded test instance (``test_N3_W4``, copy numbers [2, 1, 1]).  Builds
the HUBO Hamiltonian, constructs a grid ``ExtendedSwapStrategy``, and runs the
pass manager pipeline with a fixed SWAP-layer budget (``layer = 10``) using a
trivial identity layout.

Intended for interactive profiling (e.g. cProfile or line_profiler) rather
than production use.  No output is written to disk.
"""
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate, PauliEvolutionGate


from qiskit.transpiler import PassManager, Layout
from qiskit.transpiler.passes import InverseCancellation, CommutativeCancellation
from qopt_best_practices.transpilation.swap_cancellation_pass import SwapToFinalMapping

from qiskit_qaoa.utils.transpiler_passes import ExtendedSwapStrategy, FindCommutingPauliEvolutionsMulti
from qiskit_qaoa.utils.commuting_gate_router_precompute import CommutingGateRouterPrecompute
from qiskit_qaoa.hubo.graph_to_hubo_hamiltonian import graph_to_hubo_hamiltonian
from qiskit_qaoa.utils.gfa_utils import gfa_file_to_graph


filename = 'test_N3_W4'
copy_numbers = [2,1,1]
layer = 10


filepath = f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/data/{filename}.gfa'
graph, n, V, T = gfa_file_to_graph(filepath, copy_numbers)
hamiltonian = graph_to_hubo_hamiltonian(graph, n, T, lamda=10, constraint_terms=1.0)
ess = ExtendedSwapStrategy.from_grid(n, T)

num_physical_qubits = ess._num_vertices
donor_qc = QuantumCircuit(num_physical_qubits)
layout = Layout({donor_qc.qubits[i]: i for i in range(num_physical_qubits)})

qc = QuantumCircuit(num_physical_qubits)
qc.append(PauliEvolutionGate(hamiltonian), [layout.get_virtual_bits()[donor_qc.qubits[i]] for i in range(num_physical_qubits)])


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

tqc = pm.run(qc)   
