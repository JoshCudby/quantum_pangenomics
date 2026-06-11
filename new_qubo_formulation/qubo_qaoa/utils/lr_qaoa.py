"""Linear-ramp QAOA parameter schedule and circuit construction.

This module implements the non-variational linear-ramp (LR) QAOA schedule, which
avoids the expensive variational optimisation loop by fixing the QAOA parameters
according to a linear ramp:

    β_j = Δβ · (1 − (j − 0.5) / p)   (mixer angles, decreasing)
    γ_j = Δγ · (j − 0.5) / p          (cost angles, increasing)

for layer index j ∈ {1, …, p}.  The two public entry points are:

* ``get_LR_qaoa_circuit`` -- builds a simulator-friendly parametrised circuit
  and binds the LR parameters.
* ``get_hardware_LR_qaoa_circuit`` -- builds the same circuit with swap-network
  routing, qubit-layout remapping, and final backend transpilation for a real
  IBM device.

Both functions optionally accept warm-start angles ``phis`` (rotation angles
per qubit for the initialisation layer and the warm-start mixer) that are
produced by the iterative Boltzmann refinement in ``iterative_qaoa_utils``.
"""
import numpy as np
from typing import Optional

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import QAOAAnsatz, PauliEvolutionGate
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_ibm_runtime import IBMBackend

from qopt_best_practices.transpilation import qaoa_swap_strategy_pm

from qubo_qaoa.utils.swap_strategy import QUBOSwapStrategy

from qiskit_qaoa.utils.logging import get_logger
logger = get_logger(__name__)


def print_circuit_info(qc, circuit_name):
    """Log the 2-qubit gate count and 2-qubit depth of a circuit.

    Args:
        qc: The Qiskit ``QuantumCircuit`` to inspect.
        circuit_name: Human-readable label included in the log message.
    """
    logger.info(f'{circuit_name} has {qc.count_ops().get("cz", 0) + qc.count_ops().get("rzz", 0) + qc.count_ops().get("cx", 0) + qc.count_ops().get("ecr", 0)} 2Q gates \
    and {qc.depth(lambda instr: len(instr.qubits) > 1)} 2Q depth')
    logger.info(qc.count_ops())


def _init_layer(num_qubits, phis: Optional[ParameterVector]):
    """Build the QAOA initialisation layer.

    Applies either warm-start ``RY(phis[i])`` rotations or uniform Hadamards
    depending on whether ``phis`` is supplied.

    Args:
        num_qubits: Number of qubits in the circuit.
        phis: A ``ParameterVector`` of length ``num_qubits`` containing the
            warm-start rotation angles.  Pass ``None`` for the standard
            equal-superposition initialisation (H on every qubit).

    Returns:
        A ``QuantumCircuit`` containing only the initialisation gates, with
        ``num_qubits`` classical bits allocated for later measurement.

    Raises:
        Exception: If ``phis`` is provided but ``len(phis) != num_qubits``.
    """
    init = QuantumCircuit(num_qubits, num_qubits)
    if phis is not None:
        if not len(phis) == num_qubits:
            raise Exception(f'Wrong number of phis, expected {num_qubits}, got {len(phis)}')
        for i in range(num_qubits):
            init.ry(phis[i], i)
    else:
        init.h(range(num_qubits))  
    return init
        

def _mixer_layer(num_qubits, phis: Optional[ParameterVector]):
    """Build a single warm-start or standard X-mixer layer.

    The layer contains a single free ``Parameter("β")``.  With warm-start
    angles the mixer is ``RY(-phis[i]) · RZ(-2β) · RY(phis[i])`` per qubit
    (a rotation about the warm-start axis); without warm-start angles it
    reduces to the standard ``RX(-2β)`` transverse-field mixer.

    Args:
        num_qubits: Number of qubits.
        phis: Warm-start angles of length ``num_qubits``, or ``None`` for the
            standard X-mixer.

    Returns:
        A ``QuantumCircuit`` with one unbound ``Parameter("β")``.
    """
    mixer = QuantumCircuit(num_qubits)
    beta = Parameter("β")
    if phis is not None:
        for i in range(num_qubits):
            mixer.ry(-phis[i], i)
            mixer.rz(-2 * beta, i)
            mixer.ry(phis[i], i)
    else:
        mixer.rx(-2 * beta, range(num_qubits))
    return mixer


def _hardware_circuit_construction(
    num_virtual_qubits: int,
    cost_op: SparsePauliOp,
    sat_map: dict,
    p: int,
    backend: Optional[IBMBackend],
    edge_colouring,
    swap_strategy: QUBOSwapStrategy,
    phis: Optional[ParameterVector]
):
    """Assemble a hardware-native p-layer QAOA circuit with swap routing.

    Splits ``cost_op`` into single-qubit (Z) and two-qubit (ZZ) terms,
    routes the ZZ terms through the ``swap_strategy`` using
    ``qaoa_swap_strategy_pm``, captures the resulting qubit permutation, and
    composes the full QAOA circuit with alternating forward / reversed cost
    layers to cancel accumulated SWAP overhead.  Mixer layers track the
    permutation so that warm-start rotations are applied to the physically
    correct qubits after each swap round.  Measurements are placed according
    to the final qubit layout.  The assembled circuit is then compiled with
    Qiskit's ``optimization_level=3`` preset pass manager for ``backend``.

    Args:
        num_virtual_qubits: Number of logical QUBO variables (qubits before
            routing).
        cost_op: The full QUBO cost Hamiltonian as a ``SparsePauliOp``.
        sat_map: Mapping ``{virtual_qubit_index: physical_qubit_index}``
            specifying which physical qubits carry QUBO variables.
        p: Number of QAOA layers.
        backend: IBM backend used for the final transpilation pass.  If
            ``None`` the function still assembles the abstract circuit.
        edge_colouring: Edge colouring dict passed to ``qaoa_swap_strategy_pm``
            to partition ZZ interactions into parallel sets.
        swap_strategy: ``QUBOSwapStrategy`` instance describing the hardware
            topology and swap-layer schedule.
        phis: Warm-start rotation angles (length ``num_virtual_qubits``), or
            ``None`` for equal-superposition initialisation.

    Returns:
        A backend-compiled ``QuantumCircuit`` with all parameters bound to
        the p-vector ``ParameterVector("γ", p)`` and ``ParameterVector("β", p)``.

    Raises:
        Exception: If ``phis`` is provided but its length differs from
            ``num_virtual_qubits``.
    """
    print('Constructing circuit')
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
        # "basis_gates": ["rz", "rzz", "cz", "x", "swap"]
    }
    
    properties = {}
    def get_permutation(pass_, dag, time, property_set, count):
        properties["virtual_permutation_layout"] = property_set["virtual_permutation_layout"]
    pm = qaoa_swap_strategy_pm(config)
    print('Transpiling doubles')
    tdoubles_circ: QuantumCircuit = pm.run(doubles_circ, callback=get_permutation)
    
    singles_circ = QuantumCircuit(n)
    singles_circ.append(PauliEvolutionGate(singles, time=tdoubles_circ.parameters[0]), range(n))
    tsingles: QuantumCircuit = transpile(singles_circ, basis_gates=["rz"])
    cost_circ: QuantumCircuit = tsingles.compose(tdoubles_circ, inplace=False)
    
    init_state = QuantumCircuit(n)
    if phis is not None:
        if not len(phis) == num_virtual_qubits:
            raise Exception(f'Wrong number of phis, expected {num_virtual_qubits}, got {len(phis)}')
        for i in range(num_virtual_qubits):
            qubit = sat_map[i]
            init_state.ry(phis[i], qubit)
    else:
        init_state.h([x for x in sat_map.values()])  
        
        
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
        for i in range(num_virtual_qubits):
            qubit = sat_map[i]
            mixer_layer_odd.ry(-phis[i], qubit)
            mixer_layer_odd.rz(-2 * beta, qubit)
            mixer_layer_odd.ry(phis[i], qubit)
    else:
        mixer_layer_odd.rx(-2 * beta, [x for x in sat_map.values()])
    
    gammas = ParameterVector("γ",p)
    betas = ParameterVector("β", p)

    qaoa_circuit = QuantumCircuit(n, num_virtual_qubits)

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
            
    print('Transpiling circuit')
    generic_pm = generate_preset_pass_manager(optimization_level=3, backend=backend, scheduling_method="alap")
    backend_circ = generic_pm.run(qaoa_circuit)
        
    return backend_circ


def get_LR_qaoa_circuit(
    p: int,
    delta_b: float,
    delta_g: float,
    num_qubits: int,
    hamiltonian: SparsePauliOp,
    qaoa_circ: Optional[QuantumCircuit],
    phis: Optional[ParameterVector],
    measure: Optional[bool]
) -> tuple[QuantumCircuit, ...]:
    """Build and bind a simulator-ready LR-QAOA circuit.

    Computes the linear-ramp parameter schedule:

        β_j = Δβ · (1 − (j − 0.5) / p)
        γ_j = Δγ · (j − 0.5) / p,   j = 1, …, p

    If ``qaoa_circ`` is not provided, constructs a fresh p-layer QAOA circuit
    from ``hamiltonian``.  The cost layer is built via ``PauliEvolutionGate``
    and transpiled to a single-parameter form; the mixer is either warm-start
    (if ``phis`` is given) or the standard X-mixer.  Both the
    parameter-bound circuit and the reusable template are returned so that
    subsequent iterations can rebind parameters without rebuilding the circuit.

    Args:
        p: Number of QAOA layers.
        delta_b: Mixer amplitude Δβ; the j-th mixer angle is
            ``Δβ · (1 − (j − 0.5) / p)``.
        delta_g: Cost amplitude Δγ; the j-th cost angle is
            ``Δγ · (j − 0.5) / p``.
        num_qubits: Number of qubits (QUBO variables).
        hamiltonian: QUBO cost Hamiltonian as a ``SparsePauliOp``.
        qaoa_circ: Pre-built parametrised circuit template to reuse.  If
            ``None``, a new circuit is constructed.
        phis: Warm-start rotation angles of length ``num_qubits``, or ``None``
            for equal-superposition initialisation and X-mixer.
        measure: If ``True``, append ``measure_all``; if ``False``, append
            ``save_statevector`` (for Aer statevector simulation).

    Returns:
        A tuple ``(fixed_qc, circuit)`` where ``fixed_qc`` has all 2p
        parameters bound to the LR schedule and ``circuit`` is the
        unbound template for reuse in subsequent calls.
    """
    x = np.array([(j-0.5)/p for j in range(1, p+1)])
    betas = delta_b * (1-x)
    gammas = delta_g * x
    fixed_params = list(betas) + list(gammas)
    
    if qaoa_circ is None:
        init = _init_layer(num_qubits, phis)
        mixer = _mixer_layer(num_qubits, phis)
                    
        gamma = Parameter("γ")
        cost_circuit = QuantumCircuit(num_qubits)
        cost_circuit.append(PauliEvolutionGate(hamiltonian, time=gamma), range(num_qubits))
        cost_circuit = transpile(cost_circuit)
        
        gamma_params = ParameterVector("γ", p)
        beta_params = ParameterVector("β", p)
        
        circuit = QuantumCircuit(num_qubits, num_qubits)
        circuit.compose(init, range(num_qubits), inplace=True)
        for layer in range(p):
            bind_dict = {cost_circuit.parameters[0]: gamma_params[layer]}
            bound_cost_layer = cost_circuit.assign_parameters(bind_dict)

            bind_dict = {mixer.parameters[0]: beta_params[layer]}
            bound_mixer_layer = mixer.assign_parameters(bind_dict)
            
            circuit.compose(bound_cost_layer, range(num_qubits), inplace=True)
            circuit.compose(bound_mixer_layer, range(num_qubits), inplace=True)
        

        if measure:
            circuit.measure_all(add_bits=False)
        else:
            circuit.save_statevector()
        
        logger.info(f'p = {p}. Circuit depth: {circuit.depth()}')
    else:
        circuit = qaoa_circ
        
    fixed_param_bind = {circuit.parameters[i]: fixed_params[i] for i in range(2*p)}
    fixed_qc = circuit.assign_parameters(fixed_param_bind)
    return fixed_qc, circuit


def get_hardware_LR_qaoa_circuit(
    p: int,
    delta_b: float,
    delta_g: float,
    num_virtual_qubits: int,
    remapped_cost_op: SparsePauliOp,
    sat_map: dict[int, int],
    backend: IBMBackend,
    edge_colouring: dict[tuple[int, int], int],
    swap_strategy: QUBOSwapStrategy,
    qaoa_circ: Optional[QuantumCircuit],
    phis: Optional[ParameterVector],
) -> tuple[QuantumCircuit, ...]:
    """Build and bind a hardware-native LR-QAOA circuit for a real IBM backend.

    Applies the same linear-ramp schedule as ``get_LR_qaoa_circuit`` but
    delegates circuit assembly to ``_hardware_circuit_construction``, which
    performs swap routing via ``QUBOSwapStrategy`` and compiles the result
    for ``backend``.  If ``qaoa_circ`` is supplied the construction step is
    skipped and the existing template is rebound directly.

    Args:
        p: Number of QAOA layers.
        delta_b: Mixer amplitude Δβ.
        delta_g: Cost amplitude Δγ.
        num_virtual_qubits: Number of logical QUBO variables.
        remapped_cost_op: QUBO cost Hamiltonian after qubit-index remapping to
            match the physical layout.
        sat_map: Mapping ``{virtual_qubit_index: physical_qubit_index}``
            indicating which physical qubits carry QUBO variables.
        backend: Target IBM backend for final transpilation.
        edge_colouring: Edge colouring dict partitioning ZZ interactions for
            the swap-strategy pass manager.
        swap_strategy: ``QUBOSwapStrategy`` instance for the hardware topology.
        qaoa_circ: Reusable parametrised circuit template, or ``None`` to
            trigger fresh construction.
        phis: Warm-start rotation angles of length ``num_virtual_qubits``, or
            ``None`` for equal-superposition initialisation.

    Returns:
        A tuple ``(fixed_qc, circuit)`` where ``fixed_qc`` has all 2p
        parameters bound to the LR schedule and ``circuit`` is the
        (backend-compiled) template for reuse.
    """
    x = np.array([(j-0.5)/p for j in range(1, p+1)])
    betas = delta_b * (1-x)
    gammas = delta_g * x
    fixed_params = list(betas) + list(gammas)
    
    if qaoa_circ is None:
        circuit = _hardware_circuit_construction(
            num_virtual_qubits, remapped_cost_op, sat_map, p, backend, edge_colouring, swap_strategy, phis
        )
        
    else:
        circuit = qaoa_circ
        
    fixed_param_bind = {circuit.parameters[i]: fixed_params[i] for i in range(2*p)}
    fixed_qc = circuit.assign_parameters(fixed_param_bind)
    return fixed_qc, circuit