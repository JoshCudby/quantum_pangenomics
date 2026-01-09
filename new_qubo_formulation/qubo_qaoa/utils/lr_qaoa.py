import numpy as np
from typing import Optional

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import QAOAAnsatz

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
            mixer.rx(-2 * beta, range(num_qubits))
        

        circuit = QAOAAnsatz(hamiltonian, reps=p, initial_state=init, mixer_operator=mixer, flatten=True)
        t_circuit = transpile(circuit)
        if measure:
            t_circuit.measure_all()
        else:
            t_circuit.save_statevector()
        
        logger.info(f'p = {p}. Circuit depth: {t_circuit.depth()}')
    else:
        t_circuit = qaoa_circ
        
    fixed_param_bind = {t_circuit.parameters[i]: fixed_params[i] for i in range(2*p)}
    fixed_qc = t_circuit.assign_parameters(fixed_param_bind)
    return fixed_qc, t_circuit