import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer.primitives import SamplerV2 as Sampler
from qiskit.quantum_info import Statevector
from .logging import get_logger


logger = get_logger(__name__)

def sample_optimized_circuit(
    circuit: QuantumCircuit,
    optimized_params: np.ndarray,
    sampler: Sampler,
    shots: int
):
    optimized_circuit = circuit.assign_parameters(optimized_params)
    optimized_circuit.measure_all()
    pub= (optimized_circuit, )

    logger.info('About to sample')
    job = sampler.run(pubs=[pub], shots=shots)
    logger.info('Sampling finished')
    
    counts_bin = job.result()[0].data.meas.get_counts()
    final_distribution_bin = {key: val/shots for key, val in counts_bin.items()}
    return final_distribution_bin


def get_optimized_circuit_probabilities(
        circuit: QuantumCircuit,
        optimized_params: np.ndarray
):
    optimized_circuit = circuit.assign_parameters(optimized_params)
    optimized_circuit.remove_final_measurements() 
    statevector = Statevector(optimized_circuit)
    return np.abs(statevector.data) ** 2
