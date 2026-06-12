"""Preset PassManager pipelines for HUBO and standard QAOA compilation.

Provides two factory functions that assemble Qiskit ``PassManager`` pipelines
tailored to different QAOA problem types:

- ``get_hubo_pass_manager``: the default pipeline for Higher-Order Unconstrained
  Binary Optimisation (HUBO) problems; uses ``CommutingGateRouterPrecomputeRzz``
  with precomputed swap sequences and native RZZ gate output.
- ``get_optimal_pass_manager``: a standard pipeline for quadratic QUBO problems
  on a line topology; uses Qiskit's built-in ``Commuting2qGateRouter`` and
  assembles the full staged pass manager for a real backend.
"""

from qiskit.circuit.library import CXGate

from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import InverseCancellation, CommutativeCancellation
from qopt_best_practices.transpilation.swap_cancellation_pass import SwapToFinalMapping


from qiskit_qaoa.utils.transpiler_passes import ExtendedSwapStrategy, FindCommutingPauliEvolutionsMulti
from qiskit_qaoa.utils.commuting_gate_router import CommutingGateRouter
from qiskit_qaoa.utils.commuting_gate_router_precompute_rzz import CommutingGateRouterPrecomputeRzz


def get_hubo_pass_manager(
    extended_swap_strat: ExtendedSwapStrategy,
    max_layers: int,
    perform_extra_swaps: bool
) -> PassManager:
    """Build the default PassManager for HUBO QAOA compilation.

    The pipeline:

    1. ``FindCommutingPauliEvolutionsMulti`` — groups commuting Pauli
       evolutions into ``CommutingBlock`` gates.
    2. ``CommutingGateRouterPrecomputeRzz`` — routes the blocks using
       precomputed swap sequences and emits native RZZ gates.
    3. ``SwapToFinalMapping`` — absorbs trailing SWAPs into the output layout.
    4. ``InverseCancellation`` — cancels adjacent CX pairs.
    5. ``CommutativeCancellation`` — further simplifies by commuting and
       cancelling gates across the standard basis.
    6. ``InverseCancellation`` — final CX cancellation pass.

    Args:
        extended_swap_strat: The ``ExtendedSwapStrategy`` that defines the
            hardware topology and precomputed swap layers.
        max_layers: Maximum number of swap layers the router is allowed to
            consume per QAOA cost layer.
        perform_extra_swaps: If ``True``, interactions that cannot be routed
            within ``max_layers`` are compiled separately using Qiskit's preset
            pass manager and appended to the circuit.

    Returns:
        A configured ``PassManager`` ready to accept a QAOA cost-layer circuit.
    """
    return PassManager(
        [
            FindCommutingPauliEvolutionsMulti(), 
            CommutingGateRouterPrecomputeRzz(
                extended_swap_strat,
                max_layers=max_layers,
                perform_extra_swaps=perform_extra_swaps                
            ),
            SwapToFinalMapping(),
            InverseCancellation(gates_to_cancel=[CXGate()]),
            CommutativeCancellation(basis_gates=["cx", "swap", "rz"]),
            InverseCancellation(gates_to_cancel=[CXGate()]),
        ]
    )