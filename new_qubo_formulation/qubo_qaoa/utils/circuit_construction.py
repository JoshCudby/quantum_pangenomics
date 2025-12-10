from typing import Optional

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz, PauliEvolutionGate
from qiskit.circuit import Parameter,ParameterVector
from qiskit_ibm_runtime import IBMBackend
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qopt_best_practices.transpilation import qaoa_swap_strategy_pm

from qubo_qaoa.utils.swap_strategy import ExtendedSwapStrategy


def circuit_construction(
    num_qubits: int,
    cost_op: SparsePauliOp,
    sat_map: dict,
    p: int,
    backend: Optional[IBMBackend],
    edge_colouring,
    swap_strategy: ExtendedSwapStrategy,
) -> dict[str, QuantumCircuit]:
    circuits_dict = {}    
    n = swap_strategy._num_vertices
    

    singles = cost_op[cost_op.paulis.z.sum(axis=-1) == 1]
    doubles = cost_op[cost_op.paulis.z.sum(axis=-1) == 2]

    doubles_circ = QAOAAnsatz(
        doubles,
        initial_state=QuantumCircuit(n),
        mixer_operator=QuantumCircuit(n)
    )
    config = {
        "num_layers": 1,
        "swap_strategy": swap_strategy,
        "edge_coloring": edge_colouring,
        "construct_qaoa": False,
        "basis_gates": ["rz", "cx", "swap"]
    }
    properties = {}
    def get_permutation(pass_, dag, time, property_set, count):
        properties["virtual_permutation_layout"] = property_set["virtual_permutation_layout"]
    pm = qaoa_swap_strategy_pm(config)
    tdoubles_circ = pm.run(doubles_circ, callback=get_permutation)
    singles_circ = QuantumCircuit(n)
    singles_circ.append(PauliEvolutionGate(singles, time=tdoubles_circ.parameters[0]), range(n))
    tsingles = transpile(singles_circ, basis_gates=["rz"])
    cost_circ: QuantumCircuit = tsingles.compose(tdoubles_circ, inplace=False)

    mixer_layer_even = QuantumCircuit(n)
    beta = Parameter("β")
    mixer_layer_even.rx(2 * beta, [properties['virtual_permutation_layout'].get_physical_bits()[x] for x in sat_map.values()])

    mixer_layer_odd = QuantumCircuit(n)
    beta = Parameter("β")
    mixer_layer_odd.rx(2 * beta, [x for x in sat_map.values()])
    
    gammas = ParameterVector("γ",p)
    betas = ParameterVector("β", p)

    qaoa_circuit = QuantumCircuit(n, num_qubits)

    init_state = QuantumCircuit(n)
    init_state.h([x for x in sat_map.values()])


    qaoa_circuit.compose(init_state, inplace=True)
    for layer in range(p):
        bind_dict = {cost_circ.parameters[0]: gammas[layer]}
        bound_cost_layer = cost_circ.assign_parameters(bind_dict)

        mixer_layer = mixer_layer_even if p % 2 == 0 else mixer_layer_odd
        bind_dict = {mixer_layer.parameters[0]: betas[layer]}
        bound_mixer_layer = mixer_layer.assign_parameters(bind_dict)

        if layer % 2 == 0:
            # even layer -> append cost
            qaoa_circuit.compose(bound_cost_layer, range(n), inplace=True)
        else:
            # odd layer -> append reversed cost
            qaoa_circuit.compose(
                bound_cost_layer.reverse_ops(), range(n), inplace=True
            )

        qaoa_circuit.compose(bound_mixer_layer, range(n), inplace=True)

    if p % 2 == 1:
        # iterate over layout permutations to recover measurements
        if properties["virtual_permutation_layout"]:
            inv_sat_map = {v: k for k, v in sat_map.items()}
            for cidx, qidx in [(inv_sat_map[x], properties['virtual_permutation_layout'].get_physical_bits()[x]) for x in sat_map.values()]:
                qaoa_circuit.measure(qidx, cidx)
        else:
            print("layout not found, assigining trivial layout")
            for cidx, qidx in sat_map.items():
                qaoa_circuit.measure(qidx, cidx)
    else:
        for cidx, qidx in sat_map.items():
            qaoa_circuit.measure(qidx, cidx)
            
            
    circuits_dict["circuit_to_sample"] = qaoa_circuit

    if backend is not None:
        generic_pm = generate_preset_pass_manager(optimization_level=3, backend=backend, scheduling_method="alap")
        circuits_dict["backend"] = generic_pm.run(qaoa_circuit)
        
    return circuits_dict