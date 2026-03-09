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