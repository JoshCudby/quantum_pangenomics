import numpy as np
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import SamplerV2 as Sampler


def sample_optimized_circuit(
        backend,
        circuit: QuantumCircuit,
        optimized_params: np.ndarray
):
    sampler = Sampler(mode=backend)
    optimized_circuit = circuit.assign_parameters(optimized_params)
    pub= (optimized_circuit, )
    job = sampler.run([pub], shots=int(min(10 * 2 ** optimized_circuit.num_qubits, 1e7)))
    counts_int = job.result()[0].data.meas.get_int_counts()
    counts_bin = job.result()[0].data.meas.get_counts()
    shots = sum(counts_int.values())
    final_distribution_bin = {key: val/shots for key, val in counts_bin.items()}
    return final_distribution_bin
