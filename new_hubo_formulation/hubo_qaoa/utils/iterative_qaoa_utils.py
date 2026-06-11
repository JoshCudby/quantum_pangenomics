"""Boltzmann warm-start iterative QAOA utilities for HUBO circuits.

Implements the iterative Boltzmann warm-start procedure adapted for HUBO QAOA.
At each iteration, bitstring samples from the current circuit are weighted by a
Boltzmann factor ``exp(−β_T · E²)`` to compute per-qubit biases, which are then
converted to new Ry rotation angles for the next iteration's initial state.

Compared to the QUBO iterative QAOA:

* There is no ``ising_offset`` field in ``IterativeQAOAData`` because the HUBO
  Hamiltonian is already expressed purely in terms of Pauli-Z products with no
  constant shift that needs to be tracked separately.
* The default number of iterations is smaller (5 vs 10) to account for the higher
  cost of simulating larger HUBO circuits.
* The Boltzmann inverse temperature ``β_T`` follows a quadratic ramp rather than a
  linear one.
"""

import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import Sampler
from qiskit_aer.primitives import SamplerV2

from qiskit_qaoa.utils.string_utils import evaluate_sparse_pauli_samples

@dataclass()
class IterativeQAOAData:
    """Immutable hyperparameters for a single iterative HUBO QAOA run.

    Passed unchanged to every call of ``iteration`` so that global settings do not
    need to be threaded through the call stack.

    Note: Unlike the QUBO ``IterativeQAOAData``, this dataclass has no
    ``ising_offset`` field.  The HUBO Hamiltonian has no constant diagonal shift
    that must be removed before energy comparisons.

    Attributes:
        hamiltonian: The normalised HUBO cost Hamiltonian (``SparsePauliOp``) used
            to evaluate bitstring energies via ``evaluate_sparse_pauli_samples``.
        eta: Warm-start bias direction; must be ``+1`` or ``-1``.  ``+1`` biases
            qubits towards their low-energy assignment, ``-1`` biases away from it.
        eps: Clipping threshold for Ry rotation probabilities.  Probabilities are
            clamped to ``[eps, 1 − eps]`` to prevent angles collapsing to 0 or π.
        alpha: Sub-sampling fraction in ``(0, 1]``.  After sorting bitstrings by
            energy, only the lowest-energy fraction ``alpha`` of samples is used to
            compute the new warm-start angles.
    """
    hamiltonian: SparsePauliOp
    eta: int
    eps: float
    alpha: float
    

def get_beta_T(i: int, max_beta_T: float, max_iterations: int=10):
    """Return the Boltzmann inverse temperature for iteration ``i``.

    Follows a quadratic ramp:

    .. code-block:: none

        β_T(i) = ((i² / (max_iterations − 1)) + 1) / max_iterations * max_beta_T

    This increases from ``max_beta_T / max_iterations`` at ``i = 0`` to
    ``max_beta_T`` at ``i = max_iterations − 1``, sharpening the Boltzmann
    distribution gradually over the course of the warm-start.

    Args:
        i: Current iteration index (0-based).
        max_beta_T: Maximum inverse temperature, reached at the final iteration.
        max_iterations: Total number of warm-start iterations.

    Returns:
        The Boltzmann inverse temperature ``β_T`` for iteration ``i``.
    """
    # A quadratic ramp from 1/max_iterations to 1 over max_iterations iterations, scaled by max_beta_T
    return ( (i ** 2)/(max_iterations - 1) + 1 ) / max_iterations * max_beta_T

def _boltzmann(energies: npt.NDArray, beta_T: float) -> npt.NDArray:
    """Compute normalised Boltzmann weights for a set of bitstring energies.

    Uses an energy-squared exponent so that the distribution sharpens around
    ``E = 0`` (the ground state) rather than around the minimum energy:

    .. code-block:: none

        B_i = exp(−β_T · E_i²) / ∑_j exp(−β_T · E_j²)

    Args:
        energies: Array of bitstring energies.
        beta_T: Boltzmann inverse temperature.

    Returns:
        Normalised probability weights summing to 1.
    """
    B = np.exp(- beta_T * energies ** 2)
    return B / np.sum(B)

def _bias(boltzmanns: npt.NDArray, q: int, samples: list[str]) -> float:
    """Compute the Boltzmann-weighted bias of qubit ``q`` across all samples.

    The bias is the Boltzmann-weighted expectation of the ``±1`` assignment for
    qubit ``q``:

    .. code-block:: none

        bias_q = ∑_i B_i * (−1 if bit q of sample i is '1' else +1)

    A positive bias indicates the qubit tends to be in state ``|0⟩`` in
    low-energy samples; a negative bias indicates ``|1⟩``.

    Args:
        boltzmanns: Normalised Boltzmann weights, one per sample.
        q: Index of the qubit whose bias is computed (in bit-string indexing,
            where the leftmost character is qubit 0).
        samples: List of measurement outcome strings (most-significant bit first).

    Returns:
        The Boltzmann-weighted bias for qubit ``q``, in the range ``[−1, 1]``.
    """
    return sum([boltzmanns[i] * (-1 if samples[i][q] == '1' else 1) for i in range(len(samples))])

def _get_biases(
    samples: list[str],
    energies: npt.NDArray,
    beta_T: float
) -> npt.NDArray:
    """Compute per-qubit Boltzmann-weighted biases for all qubits.

    Calls ``_bias`` for each qubit position and reverses the ordering so that the
    returned array is indexed by physical qubit index (qubit 0 first) rather than
    by bit-string position (most-significant bit first).

    Args:
        samples: List of measurement outcome bit-strings.
        energies: Array of bitstring energies, aligned with ``samples``.
        beta_T: Boltzmann inverse temperature passed to ``_boltzmann``.

    Returns:
        Array of shape ``(num_qubits,)`` containing the bias for each qubit,
        indexed by physical qubit order (qubit 0 at index 0).
    """
    num_qubits = len(samples[0])
    boltzmanns = _boltzmann(energies, beta_T)
    return np.array([_bias(boltzmanns, q, samples) for q in range(num_qubits)][::-1])

def _get_angles(
    samples: list[str],
    energies: npt.NDArray,
    beta_T: float,
    eta: int,
    eps: float
) -> npt.NDArray:
    """Convert Boltzmann-weighted biases to Ry rotation angles for the next iteration.

    Maps per-qubit biases to probabilities and then to ``Ry`` angles via:

    .. code-block:: none

        p_q = 0.5 * (1 − η · bias_q)        (clipped to [eps, 1−eps])
        θ_q = 2 · arcsin(√p_q)

    The angle ``θ_q`` is the ``Ry`` rotation that prepares ``|1⟩`` with probability
    ``p_q``, used as the initial-state angle for the next warm-start iteration.

    Args:
        samples: List of measurement outcome bit-strings.
        energies: Array of bitstring energies, aligned with ``samples``.
        beta_T: Boltzmann inverse temperature.
        eta: Bias direction; must be ``+1`` or ``-1``.
        eps: Probability clipping threshold; probabilities are clamped to
            ``[eps, 1 − eps]``.

    Returns:
        Array of shape ``(num_qubits,)`` containing the Ry rotation angle for
        each qubit.

    Raises:
        Exception: If ``eta`` is not ``+1`` or ``-1``.
    """
    if eta not in [-1, 1]:
        raise Exception(f'Eta should be in [-1,1], got {eta}')
    biases = _get_biases(samples, energies, beta_T)
    probabilities = 0.5 * (1 - eta * biases)
    probabilities[probabilities < eps] = eps
    probabilities[probabilities > 1 - eps] = 1 - eps

    angles = 2 * np.arcsin(np.sqrt(probabilities))
    return angles

def _subsample(samples, energies, alpha=1.0) -> tuple[list[str], npt.NDArray]:
    """Retain the lowest-energy fraction of samples for bias computation.

    Sorts samples by energy in ascending order and keeps the first
    ``floor(alpha * N)`` entries.  This focuses the warm-start update on the
    most promising bitstrings and discards high-energy noise.

    Args:
        samples: List of measurement outcome bit-strings.
        energies: Array of bitstring energies corresponding to ``samples``.
        alpha: Fraction of samples to retain, in the range ``(0, 1]``.
            ``1.0`` retains all samples.

    Returns:
        A two-tuple ``(subsamples, subenergies)`` where both are sorted by energy
        ascending and truncated to ``floor(alpha * N)`` entries.
    """
    idx = np.argsort(energies)
    sorted_energies = energies[idx]
    sorted_samples = [samples[i] for i in idx]
    end_idx = int(alpha * len(energies))
    return sorted_samples[:end_idx], sorted_energies[:end_idx]


def iteration(
    qc: QuantumCircuit,
    sampler: Sampler | SamplerV2,
    shots: int,
    angles: npt.NDArray,
    beta_T: float,
    data: IterativeQAOAData,
    history: list
):
    """Execute one Boltzmann warm-start iteration and return updated rotation angles.

    Binds the current warm-start angles to the parameterised QAOA circuit, runs
    ``shots`` shots via the sampler, evaluates bitstring energies against the HUBO
    Hamiltonian, sub-samples the lowest-energy fraction, and computes new ``Ry``
    rotation angles for the next iteration.

    The iteration history entry appended to ``history`` has the form
    ``[samples, energies, mean_energy]``.

    Args:
        qc: Parameterised QAOA circuit whose parameters are the ``num_qubits`` Ry
            angles for the initial state.
        sampler: Qiskit ``Sampler`` or ``SamplerV2`` primitive for shot-based
            simulation or hardware execution.
        shots: Number of measurement shots per iteration.
        angles: Current Ry rotation angles, one per qubit.
        beta_T: Boltzmann inverse temperature for this iteration, obtained from
            ``get_beta_T``.
        data: Immutable hyperparameters (Hamiltonian, ``eta``, ``eps``, ``alpha``).
        history: Mutable list accumulating per-iteration records.  Each call
            appends ``[samples, energies, mean_energy]``.

    Returns:
        Updated Ry rotation angles (``npt.NDArray`` of shape ``(num_qubits,)``) for
        use in the next iteration.
    """
    sample_circuit = qc.assign_parameters(angles, inplace=False)
    sampler_job = sampler.run([sample_circuit], shots=shots)
    sampler_result = sampler_job.result()
    counts = sampler_result[0].data.c.get_counts()
    
    evals = evaluate_sparse_pauli_samples(counts.keys(), data.hamiltonian)
    samples, energies = [], []
    for idx, (sample, count) in enumerate(counts.items()):
        samples.extend(count * [sample])
        energies.extend(count * [evals[idx]])
    energies = np.array(energies)
    total_energy = np.mean(energies)
    
    subsamples, subenergies = _subsample(samples, energies, data.alpha)
    history.append([samples, energies, total_energy])
    new_angles = _get_angles(
        subsamples, 
        subenergies, 
        beta_T,
        data.eta,
        data.eps
    )
    return new_angles