"""Assembles a complete p-layer QAOA circuit with hardware swap routing.

This module provides ``circuit_construction``, which builds a parametrised
QAOA circuit suitable for both simulator use and hardware deployment on IBM
devices.  The key steps performed are:

1. **Cost-layer routing**: The QUBO cost operator is split into single-qubit
   (Z) and two-qubit (ZZ) terms.  The ZZ terms are routed through the
   provided ``QUBOSwapStrategy`` via ``qaoa_swap_strategy_pm``, and the
   resulting qubit permutation is captured for later measurement remapping.

2. **Alternating cost layers**: Even-indexed QAOA layers append the cost
   circuit in forward order; odd-indexed layers append it in reversed order
   (``reverse_ops``).  This cancels the net qubit permutation accumulated by
   the swap network over even numbers of layers.

3. **Warm-start mixer**: If warm-start angles ``phis`` are provided, the
   mixer per qubit is ``RY(-phis[i]) · RZ(-2β) · RY(phis[i])``; otherwise
   the standard ``RX(-2β)`` X-mixer is used.  After an odd number of layers
   the permutation is tracked to route mixer gates to their physically correct
   qubits.

4. **Measurement remapping**: Classical bit indices are assigned by inverting
   the combined ``sat_map`` and ``virtual_permutation_layout`` so that
   bitstring bit ``i`` always corresponds to QUBO variable ``i``.

5. **Optional backend compilation**: If ``backend`` is supplied, the circuit
   is further compiled with Qiskit's ``optimization_level=3`` preset pass
   manager (ALAP scheduling).

The function returns a dict with key ``"circuit_to_sample"`` (abstract
circuit) and, if a backend is supplied, ``"backend"`` (compiled circuit).
"""
from typing import Optional

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz, PauliEvolutionGate
from qiskit.circuit import Parameter,ParameterVector
from qiskit_ibm_runtime import IBMBackend
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qopt_best_practices.transpilation import qaoa_swap_strategy_pm

from qubo_qaoa.utils.swap_strategy import QUBOSwapStrategy


def circuit_construction(
    num_qubits: int,
    cost_op: SparsePauliOp,
    sat_map: dict,
    p: int,
    backend: Optional[IBMBackend],
    edge_colouring,
    swap_strategy: QUBOSwapStrategy,
    phis: Optional[ParameterVector]
) -> dict[str, QuantumCircuit]:
    """Build a parametrised p-layer QAOA circuit with hardware-aware swap routing.

    Constructs the full QAOA ansatz (initialisation + p × (cost + mixer) layers
    + measurements) for a QUBO Hamiltonian encoded as a ``SparsePauliOp``.
    Routing of the two-qubit cost terms is handled by ``QUBOSwapStrategy``;
    the resulting qubit permutation is tracked and inverted when constructing
    measurements and mixer layers.

    The returned circuits use ``ParameterVector("γ", p)`` for cost-layer
    angles and ``ParameterVector("β", p)`` for mixer angles, laid out so
    that ``circuit.parameters[0..p-1]`` are the gammas and
    ``circuit.parameters[p..2p-1]`` are the betas.

    Args:
        num_qubits: Number of logical QUBO variables.  Must equal
            ``len(sat_map)``.
        cost_op: Full QUBO cost Hamiltonian as a ``SparsePauliOp`` over
            ``swap_strategy._num_vertices`` qubits.
        sat_map: Mapping ``{virtual_qubit_index: physical_qubit_index}``
            specifying which physical qubits carry QUBO variables (ancilla /
            bridge qubits are not included).
        p: Number of QAOA layers.
        backend: IBM backend for the optional final compilation pass.  Pass
            ``None`` to skip hardware compilation.
        edge_colouring: Edge colouring dict ``{(qubit_i, qubit_j): colour}``
            partitioning ZZ interactions into parallel sets for the swap
            strategy pass manager.
        swap_strategy: ``QUBOSwapStrategy`` instance encoding the hardware
            topology and SWAP-layer schedule used to route ZZ interactions.
        phis: Warm-start rotation angles of length ``num_qubits`` as a
            ``ParameterVector``, or ``None`` for equal-superposition
            initialisation and the standard X-mixer.

    Returns:
        A dict with the following keys:

        * ``"circuit_to_sample"``: The abstract parametrised QAOA circuit
          (not yet compiled for any specific backend).
        * ``"backend"`` *(present only when* ``backend`` *is not* ``None``):
          The circuit compiled with ``optimization_level=3`` and ALAP
          scheduling for the provided backend.

    Raises:
        Exception: If ``phis`` is provided but ``len(phis) != num_qubits``.
    """
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
    if phis is not None:
        inv_sat_map = {v: k for k, v in sat_map.items()}
        for i, qidx in [(inv_sat_map[x], properties['virtual_permutation_layout'].get_physical_bits()[x]) for x in sat_map.values()]:
            mixer_layer_even.ry(-phis[i], qidx)
            mixer_layer_even.rz(-2 * beta, qidx)
            mixer_layer_even.ry(phis[i], qidx)
    else:
        mixer_layer_even.rx(-2 * beta, [properties['virtual_permutation_layout'].get_physical_bits()[x] for x in sat_map.values()])
    
    
    mixer_layer_odd = QuantumCircuit(n)
    if phis is not None:
        for i in range(num_qubits):
            qubit = sat_map[i]
            mixer_layer_odd.ry(-phis[i], qubit)
            mixer_layer_odd.rz(-2 * beta, qubit)
            mixer_layer_odd.ry(phis[i], qubit)
    else:
        mixer_layer_odd.rx(-2 * beta, [x for x in sat_map.values()])
    
    gammas = ParameterVector("γ",p)
    betas = ParameterVector("β", p)

    qaoa_circuit = QuantumCircuit(n, num_qubits)


    init_state = QuantumCircuit(n)
    if phis is not None:
        if not len(phis) == num_qubits:
            raise Exception(f'Wrong number of phis, expected {num_qubits}, got {len(phis)}')
        for i in range(num_qubits):
            qubit = sat_map[i]
            init_state.ry(phis[i], qubit)
    else:
        init_state.h([x for x in sat_map.values()])  

    qaoa_circuit.compose(init_state, inplace=True)
    for layer in range(p):
        bind_dict = {cost_circ.parameters[0]: gammas[layer]}
        bound_cost_layer = cost_circ.assign_parameters(bind_dict)

        mixer_layer = mixer_layer_even if layer % 2 == 0 else mixer_layer_odd
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