"""Linear-ramp QAOA schedule for HUBO circuits with pre-compiled cost layers.

Implements the linear-ramp (LR) QAOA angle schedule of Sack & Serbyn (2021) for
HUBO cost functions.  Unlike the QUBO version (which builds the cost circuit
on-the-fly from a Hamiltonian), this module accepts a ``cost_circuit`` that has
already been compiled and optimally routed by ``compilation.py``.  The circuit is
re-parameterised via ``parameterise_circuit`` so that a single scalar ``γ`` can be
swept, and the mixer layer is constructed separately.

The LR schedule sets:

.. code-block:: none

    x_j = (j − 0.5) / p  for j = 1, …, p
    β_j = δ_β · (1 − x_j)
    γ_j = δ_γ · x_j

Both a simulation path (``get_LR_qaoa_circuit``) and a hardware path
(``get_hardware_LR_qaoa_circuit``) are provided; the hardware path accounts for
qubit permutations introduced by SWAP routing.
"""

import numpy as np
import networkx as nx
from typing import Optional

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.transpiler import Layout
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_qaoa.utils.transpiler_passes import ExtendedSwapStrategy
from qiskit_ibm_runtime import IBMBackend

from qiskit_qaoa.utils.logging import get_logger
logger = get_logger(__name__)


def get_LR_qaoa_circuit(
    p: int,
    delta_b: float,
    delta_g: float,
    num_qubits: int,
    cost_circuit: QuantumCircuit,
    qaoa_circ: Optional[QuantumCircuit],
    phis: Optional[ParameterVector],
    measure: Optional[bool]
) -> tuple[QuantumCircuit, QuantumCircuit]:
    """Build (or bind) a linear-ramp QAOA circuit for simulation.

    Constructs a ``p``-layer QAOA circuit using the pre-compiled ``cost_circuit`` as
    the cost layer.  Unlike the QUBO version, ``cost_circuit`` is supplied externally
    and must already contain exactly one free ``Parameter`` (the cost angle ``γ``).

    If ``qaoa_circ`` is ``None`` the full abstract circuit is constructed, including:

    * An initial state layer: Hadamard gates (or ``Ry(ϕ_i)`` rotations when Boltzmann
      warm-start angles ``phis`` are provided).
    * ``p`` alternating cost and mixer layers with LR-scheduled angles.
    * A terminal ``measure_all`` or ``save_statevector`` instruction.

    If ``qaoa_circ`` is provided it is reused and only the parameter binding step is
    performed.

    Key difference from the QUBO ``lr_qaoa.py``: this function accepts ``cost_circuit``
    (a pre-compiled ``QuantumCircuit``) rather than a ``SparsePauliOp`` Hamiltonian,
    and also accepts a ``qaoa_circ`` argument to allow circuit reuse across
    warm-start iterations.

    Args:
        p: Number of QAOA layers.
        delta_b: Amplitude of the linear-ramp mixer schedule.
        delta_g: Amplitude of the linear-ramp cost schedule.
        num_qubits: Number of qubits in the circuit.
        cost_circuit: Pre-compiled, parameterised cost unitary with a single free
            ``Parameter`` (``γ``).
        qaoa_circ: If not ``None``, skip circuit construction and bind parameters
            to this existing circuit directly.
        phis: Optional Boltzmann warm-start angles.  If provided, the initial state
            is ``⊗_i Ry(ϕ_i)|0⟩`` and the mixer applies ``Ry(−ϕ_i) Rz(−2β) Ry(ϕ_i)``;
            otherwise ``H`` and ``Rx(−2β)`` are used.
        measure: If ``True``, append ``measure_all``; if ``False`` or ``None``,
            append ``save_statevector`` for statevector simulation.

    Returns:
        A two-tuple ``(fixed_qc, circuit)`` where:

        * ``fixed_qc`` (``QuantumCircuit``) – the circuit with all ``2p`` parameters
          bound to the LR-schedule values.
        * ``circuit`` (``QuantumCircuit``) – the abstract (unbound) QAOA circuit,
          useful for subsequent warm-start iterations.
    """
    x = np.array([(j-0.5)/p for j in range(1, p+1)])
    betas = delta_b * (1-x)
    gammas = delta_g * x
    fixed_params = list(betas) + list(gammas)
    
    if qaoa_circ is None:
        circuit = QuantumCircuit(num_qubits, num_qubits)
        if phis is not None:
            for i in range(num_qubits):
                circuit.ry(phis[i], i)
        else:
            circuit.h(range(num_qubits))
        
        mixer_layer = QuantumCircuit(num_qubits)
        beta = Parameter("β")
        if phis is not None:
            for i in range(num_qubits):
                mixer_layer.ry(-phis[i], i)
                mixer_layer.rz(-2*beta, i)
                mixer_layer.ry(phis[i], i)
        else:
            mixer_layer.rx(-2 * beta, range(num_qubits))
        
        gamma_params = ParameterVector("γ", p)
        beta_params = ParameterVector("β", p)
        
        for layer in range(p):
            bind_dict = {cost_circuit.parameters[0]: gamma_params[layer]}
            bound_cost_layer = cost_circuit.assign_parameters(bind_dict)

            bind_dict = {mixer_layer.parameters[0]: beta_params[layer]}
            bound_mixer_layer = mixer_layer.assign_parameters(bind_dict)
            
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


def _hardware_circuit_construction(
    num_virtual_qubits: int,
    cost_circuit: QuantumCircuit,
    layout: Layout,
    p: int,
    backend: Optional[IBMBackend],
    phis: Optional[ParameterVector]
):
    """Construct and transpile the hardware QAOA circuit, accounting for SWAP permutations.

    Builds a ``p``-layer QAOA circuit mapped onto physical qubits via ``layout``.
    Because the cost circuit contains SWAP gates that permute the qubit ordering,
    alternating even and odd layers use different mixer qubit assignments: even layers
    use the post-SWAP layout (``new_layout``) while odd layers use the original
    ``layout``.  The cost layer is applied in forward order for even layers and in
    reversed gate order for odd layers so the cumulative permutation cancels.

    Measurement assignments are similarly adjusted based on the parity of ``p``.

    The circuit is transpiled to the backend at optimisation level 3 for circuits with
    fewer than 25 qubits and level 1 otherwise.

    Args:
        num_virtual_qubits: Number of logical (virtual) qubits encoding the problem.
        cost_circuit: Pre-compiled, parameterised cost unitary (may contain SWAP
            gates for routing).
        layout: Initial virtual-to-physical qubit mapping.
        p: Number of QAOA layers.
        backend: Target ``IBMBackend`` for transpilation.
        phis: Optional Boltzmann warm-start angles; see ``get_LR_qaoa_circuit``.

    Returns:
        A two-tuple ``(backend_circ, qaoa_circuit)`` where:

        * ``backend_circ`` (``QuantumCircuit``) – the transpiled, backend-native
          circuit ready for execution.
        * ``qaoa_circuit`` (``QuantumCircuit``) – the abstract (pre-transpile)
          hardware-mapped QAOA circuit.
    """
    n = cost_circuit.num_qubits
    donor_qc = QuantumCircuit(n)
    qubits = donor_qc.qubits
    trivial_layout = Layout.generate_trivial_layout(donor_qc.qregs[0])

    
    init_state = QuantumCircuit(n)
    if phis is not None:
        if not len(phis) == num_virtual_qubits:
            raise Exception(f'Wrong number of phis, expected {num_virtual_qubits}, got {len(phis)}')
        for i in range(num_virtual_qubits):
            qubit = layout.get_virtual_bits()[qubits[i]]
            init_state.ry(phis[i], qubit)
    else:
        init_state.h([layout.get_virtual_bits()[qubits[i]] for i in range(num_virtual_qubits)])  
        
    new_layout = layout.copy()
    for gate in cost_circuit:
        if gate.operation.name == 'swap':
            physical_qubits_to_swap = [trivial_layout.get_virtual_bits()[q] for q in gate.qubits]
            new_layout.swap(*physical_qubits_to_swap)

    mixer_layer_even = QuantumCircuit(n)
    beta = Parameter("β")
    if phis is not None:
        for i in range(num_virtual_qubits):
            qubit = new_layout.get_virtual_bits()[qubits[i]]
            mixer_layer_even.ry(-phis[i], qubit)
            mixer_layer_even.rz(-2 * beta, qubit)
            mixer_layer_even.ry(phis[i], qubit)
    else:
        mixer_layer_even.rx(-2 * beta, [new_layout.get_virtual_bits()[qubits[i]] for i in range(num_virtual_qubits)])
    
    
    mixer_layer_odd = QuantumCircuit(n)
    if phis is not None:
        for i in range(num_virtual_qubits):
            qubit = layout.get_virtual_bits()[qubits[i]]
            mixer_layer_odd.ry(-phis[i], qubit)
            mixer_layer_odd.rz(-2 * beta, qubit)
            mixer_layer_odd.ry(phis[i], qubit)
    else:
        mixer_layer_odd.rx(-2 * beta, [layout.get_virtual_bits()[qubits[i]] for i in range(num_virtual_qubits)])
    
    gammas = ParameterVector("γ",p)
    betas = ParameterVector("β", p)

    qaoa_circuit = QuantumCircuit(n, num_virtual_qubits)

    qaoa_circuit.compose(init_state, inplace=True)
    for layer in range(p):
        bind_dict = {cost_circuit.parameters[0]: gammas[layer]}
        bound_cost_layer = cost_circuit.assign_parameters(bind_dict)

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
        for i in range(num_virtual_qubits):
            qubit = new_layout.get_virtual_bits()[qubits[i]]
            qaoa_circuit.measure(qubit, i)
    else:
        for i in range(num_virtual_qubits):
            qubit = layout.get_virtual_bits()[qubits[i]]
            qaoa_circuit.measure(qubit, i)
            
    
    if qaoa_circuit.num_qubits < 25:
        opt_level = 3
    else:
        opt_level = 1
        print(qaoa_circuit.num_qubits, qaoa_circuit.count_ops())
    print(f'Compiling to backend with opt {opt_level}')
    generic_pm = generate_preset_pass_manager(optimization_level=opt_level, backend=backend, scheduling_method="alap")
    backend_circ = generic_pm.run(qaoa_circuit)
        
    return backend_circ, qaoa_circuit
    

def get_hardware_LR_qaoa_circuit(
    p: int,
    delta_b: float,
    delta_g: float,
    num_virtual_qubits: int,
    cost_circuit: QuantumCircuit,
    layout: Layout,
    backend: IBMBackend,
    qaoa_circ: Optional[QuantumCircuit],
    phis: Optional[ParameterVector],
) -> tuple[QuantumCircuit, QuantumCircuit, Optional[QuantumCircuit]]:
    """Build (or bind) a linear-ramp QAOA circuit for hardware execution.

    Mirrors ``get_LR_qaoa_circuit`` but targets a physical IBM backend.  When
    ``qaoa_circ`` is ``None`` the abstract hardware circuit is constructed via
    ``_hardware_circuit_construction`` (which transpiles to the backend); otherwise
    the provided circuit is reused and only parameter binding is applied.

    Key difference from the QUBO ``lr_qaoa.py``: accepts a pre-compiled
    ``cost_circuit`` and a ``qaoa_circ`` argument for warm-start reuse, and returns
    the abstract (pre-transpile) circuit as a third return value.

    Args:
        p: Number of QAOA layers.
        delta_b: Amplitude of the linear-ramp mixer schedule.
        delta_g: Amplitude of the linear-ramp cost schedule.
        num_virtual_qubits: Number of logical qubits encoding the problem.
        cost_circuit: Pre-compiled, parameterised cost unitary with SWAP routing.
        layout: Initial virtual-to-physical qubit mapping used during compilation.
        backend: Target ``IBMBackend`` for transpilation.
        qaoa_circ: If not ``None``, reuse this already-transpiled circuit and bind
            parameters directly.
        phis: Optional Boltzmann warm-start angles.

    Returns:
        A three-tuple ``(fixed_qc, circuit, abstract_circuit)`` where:

        * ``fixed_qc`` (``QuantumCircuit``) – the transpiled circuit with all ``2p``
          parameters bound to the LR-schedule values.
        * ``circuit`` (``QuantumCircuit``) – the abstract (unbound) hardware circuit.
        * ``abstract_circuit`` (``QuantumCircuit`` or ``None``) – the pre-transpile
          abstract circuit returned by ``_hardware_circuit_construction``, or
          ``None`` if ``qaoa_circ`` was supplied.
    """
    x = np.array([(j-0.5)/p for j in range(1, p+1)])
    betas = delta_b * (1-x)
    gammas = delta_g * x
    fixed_params = list(betas) + list(gammas)
    abstract_circuit = None
    
    if qaoa_circ is None:
        circuit, abstract_circuit = _hardware_circuit_construction(
            num_virtual_qubits, cost_circuit, layout, p, backend, phis
        )
        logger.info(f'p = {p}. Circuit depth: {circuit.depth()}')
    else:
        circuit = qaoa_circ
        
    fixed_param_bind = {circuit.parameters[i]: fixed_params[i] for i in range(2*p)}
    fixed_qc = circuit.assign_parameters(fixed_param_bind)
    return fixed_qc, circuit, abstract_circuit