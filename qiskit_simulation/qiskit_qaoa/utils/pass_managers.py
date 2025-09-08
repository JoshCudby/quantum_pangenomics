from qiskit.circuit.library import CXGate

from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import InverseCancellation, CommutativeCancellation
from qopt_best_practices.transpilation.swap_cancellation_pass import SwapToFinalMapping


from qiskit_qaoa.utils.transpiler_passes import ExtendedSwapStrategy, FindCommutingPauliEvolutionsMulti
from qiskit_qaoa.utils.commuting_gate_router import CommutingGateRouter


def get_hubo_pass_manager(
    extended_swap_strat: ExtendedSwapStrategy, 
    max_layers: int, 
    perform_extra_swaps: bool
) -> PassManager:
    return PassManager(
        [
            FindCommutingPauliEvolutionsMulti(), 
            CommutingGateRouter(
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