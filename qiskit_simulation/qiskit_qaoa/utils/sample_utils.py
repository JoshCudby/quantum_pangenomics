import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer.primitives import SamplerV2 as Sampler
from .logging import get_logger

logger = get_logger(__name__)

def sample_optimized_circuit(
        circuit: QuantumCircuit,
        optimized_params: np.ndarray
):
    batched_shots_gpu=True
    batched_shots_gpu_max_qubits=30
    logger.info(f'Batched shots GPU: {batched_shots_gpu}, max qubits: {batched_shots_gpu_max_qubits}')
    sampler = Sampler(
        options=dict(backend_options=dict(
            batched_shots_gpu=batched_shots_gpu, batched_shots_gpu_max_qubits=batched_shots_gpu_max_qubits
            ))
    )

    optimized_circuit = circuit.assign_parameters(optimized_params)
    pub= (optimized_circuit, )
    shots=int(min(10 * 2 ** optimized_circuit.num_qubits, 1e7))

    logger.info('About to sample')
    job = sampler.run(pubs=[pub], shots=shots)
    logger.info('Sampling finished')
    
    counts_bin = job.result()[0].data.meas.get_counts()
    final_distribution_bin = {key: val/shots for key, val in counts_bin.items()}
    return final_distribution_bin
