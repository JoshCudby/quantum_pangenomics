import numpy as np
from typing import Optional

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import QAOAAnsatz, PauliEvolutionGate

from qiskit_qaoa.utils.logging import get_logger
logger = get_logger(__name__)


def get_LR_qaoa_circuit(
    p: int, 
    delta_b: float, 
    delta_g: float, 
    num_qubits: int,
    hamiltonian: SparsePauliOp,
    qaoa_circ: Optional[QuantumCircuit],
    phis: Optional[ParameterVector],
    measure: Optional[bool]
) -> tuple[QuantumCircuit, QuantumCircuit]:
    x = np.array([(j-0.5)/p for j in range(1, p+1)])
    betas = delta_b * (1-x)
    gammas = delta_g * x
    fixed_params = list(betas) + list(gammas)
    
    if qaoa_circ is None:
        init = QuantumCircuit(num_qubits, num_qubits)
        if phis is not None:
            for i in range(num_qubits):
                init.ry(phis[i], i)
        else:
            init.h(range(num_qubits))
        
        mixer = QuantumCircuit(num_qubits)
        beta = Parameter("β")
        if phis is not None:
            for i in range(num_qubits):
                mixer.ry(-phis[i], i)
                mixer.rz(-2 * beta, i)
                mixer.ry(phis[i], i)
        else:
            mixer.rx(-2 * beta, range(num_qubits))#
            
        gamma_params = ParameterVector("γ", p)
        beta_params = ParameterVector("β", p)
        
        gamma = Parameter("γ")
        cost_circuit = QuantumCircuit(num_qubits)
        cost_circuit.append(PauliEvolutionGate(hamiltonian, time=gamma), range(num_qubits))
        cost_circuit = transpile(cost_circuit)
        
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