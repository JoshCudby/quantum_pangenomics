from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    BasisTranslator,
    UnrollCustomDefinitions,
    CommutativeCancellation,
    HighLevelSynthesis,
    InverseCancellation,
)
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import (
    SwapStrategy,
    FindCommutingPauliEvolutions,
    Commuting2qGateRouter,
)
from qiskit.circuit.library.standard_gates.equivalence_library import _sel
from qiskit.circuit.library import CXGate
from qiskit_qaoa.utils.qaoa_pass import QAOAPass


def get_optimal_pass_manager(num_qubits, backend, initial_layout, betas):
    # 1. choose swap strategy (in this case -> line)
    swap_strategy = SwapStrategy.from_line([i for i in range(num_qubits)])
    edge_coloring = {(idx, idx + 1): (idx + 1) % 2 for idx in range(num_qubits)}

    # 2. define pass manager for cost layer
    pre_init = PassManager(
                [HighLevelSynthesis(basis_gates=['PauliEvolution']),
                FindCommutingPauliEvolutions(),
                Commuting2qGateRouter(
                        swap_strategy,
                        edge_coloring,
                    ),
                HighLevelSynthesis(basis_gates=["x", "cx", "sx", "rz", "id"]),
                InverseCancellation(gates_to_cancel=[CXGate()]),
                ]
    )

    init = PassManager([QAOAPass(num_layers=3, num_qubits=10, betas=betas)])


    # The post init step unrolls the gates in the ansatz to the backend basis gates
    post_init = PassManager(
        [
            UnrollCustomDefinitions(_sel, basis_gates=backend.operation_names),
            BasisTranslator(_sel, target_basis=backend.operation_names),
        ]
    )

    # The optimization step performs additional gate cancellations
    optimization = PassManager(
        [
        CommutativeCancellation(target=backend.target)
        ]
    )
    staged_pm = generate_preset_pass_manager(3, backend, initial_layout=initial_layout)
    staged_pm.pre_init = pre_init
    staged_pm.init = init
    staged_pm.post_init = post_init
    # staged_pm.optimization = optimization
    return staged_pm