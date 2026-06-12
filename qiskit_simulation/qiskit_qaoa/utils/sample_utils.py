"""Circuit sampling and probability distribution utilities.

Wraps Qiskit's Aer sampler and statevector simulator to provide convenient
helpers for obtaining bitstring probability distributions from an optimised
QAOA circuit.
"""

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
    """Bind optimised parameters to a circuit and sample the resulting state.

    Args:
        circuit: A parameterised QAOA ansatz circuit without measurements.
        optimized_params: A 1-D numpy array of real values to bind to the
            circuit's free parameters (gammas then betas, or as ordered by
            ``circuit.parameters``).
        sampler: An Aer ``SamplerV2`` instance configured for the desired
            noise model and number of shots.
        shots: Number of measurement shots to take.

    Returns:
        A dict mapping binary bitstring keys (str) to normalised empirical
        probabilities (float), i.e. ``count / shots``.
    """
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
    """Compute exact state-vector probabilities for an optimised circuit.

    Binds parameters, removes any final measurement gates, and simulates the
    resulting unitary exactly using Qiskit's ``Statevector`` simulator.

    Args:
        circuit: A parameterised QAOA ansatz circuit, optionally with
            measurement operations that will be stripped before simulation.
        optimized_params: A 1-D numpy array of real parameter values to bind
            to the circuit.

    Returns:
        A 1-D numpy array of length ``2**n`` containing the squared-amplitude
        probability of each computational-basis state.
    """
    optimized_circuit = circuit.assign_parameters(optimized_params)
    optimized_circuit.remove_final_measurements() 
    statevector = Statevector(optimized_circuit)
    return np.abs(statevector.data) ** 2