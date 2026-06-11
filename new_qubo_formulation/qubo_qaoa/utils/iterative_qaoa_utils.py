"""Boltzmann-weighted warm-start refinement for QAOA.

This module implements the iterative warm-start (Iter-QAOA) procedure.  After
each QAOA sampling round the measured bitstrings are weighted by a Boltzmann
distribution over their energies, yielding per-qubit bias estimates.  These
biases are converted into new single-qubit rotation angles ``ϕ_i`` via

    P_i = 0.5 · (1 − η · bias_i)
    ϕ_i = 2 · arcsin(√P_i)

where η ∈ {-1, +1} controls the sign convention.  The angles are fed back into
the initialisation layer and warm-start mixer of the next QAOA run, steering
the quantum state towards the low-energy sector.

The inverse temperature β_T grows quadratically across iterations via
``get_beta_T``, sharpening the Boltzmann weight as the refinement converges.

Typical usage::

    data = IterativeQAOAData(hamiltonian=..., ising_offset=..., eta=1,
                             eps=0.05, alpha=1.0)
    angles = np.pi / 2 * np.ones(num_qubits)
    history = []
    for i in range(max_iterations):
        beta_T = get_beta_T(i, max_beta_T)
        angles = iteration(fixed_qc, sampler, shots, angles, beta_T, data, history)
"""

import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import Sampler
from qiskit_aer.primitives import SamplerV2

from qiskit_qaoa.utils.string_utils import evaluate_sparse_pauli_samples
from qubo_qaoa.utils.postprocess import postprocess

from typing import Optional

@dataclass()
class IterativeQAOAData:
    """Global hyperparameters for an Iter-QAOA run.

    Attributes:
        hamiltonian: The QUBO cost Hamiltonian as a ``SparsePauliOp``.
            Energies are computed as ``<hamiltonian> + ising_offset``.
        ising_offset: Constant offset added to Pauli-evaluation energies to
            recover the original QUBO objective value from the Ising encoding.
        eta: Sign convention for the per-qubit bias, must be +1 or -1.
            ``eta=1`` biases towards low-energy (bit=0) states; ``eta=-1``
            inverts the bias direction.
        eps: Probability clamp applied after computing per-qubit marginal
            probabilities.  Probabilities outside ``[eps, 1-eps]`` are clipped
            to avoid ``arcsin`` domain errors.  Typical value: 0.05.
        alpha: Top-fraction parameter for subsampling.  Only the lowest-energy
            ``alpha`` fraction of shots is used when computing Boltzmann
            biases.  ``alpha=1.0`` uses all shots.
    """
    hamiltonian: SparsePauliOp
    ising_offset: float
    eta: int
    eps: float
    alpha: float
    

def get_beta_T(i: int, max_beta_T: float, max_iterations: int=10):
    """Compute the Boltzmann inverse temperature for iteration ``i``.

    The schedule grows quadratically from roughly ``max_beta_T / max_iterations``
    at iteration 0 to ``max_beta_T`` at iteration ``max_iterations - 1``:

        β_T(i) = ((i² / (max_iterations − 1)) + 1) / max_iterations · max_β_T

    Increasing β_T sharpens the Boltzmann weight over iterations, progressively
    concentrating the warm-start distribution around low-energy samples.

    Args:
        i: Current iteration index (0-based).
        max_beta_T: Maximum inverse temperature reached at the last iteration.
        max_iterations: Total number of warm-start iterations.  Defaults to 10.

    Returns:
        The inverse temperature β_T for iteration ``i``.
    """
    # A quadratic ramp from 1/max_iterations to 1 over max_iterations iterations, scaled by max_beta_T
    return ((i ** 2)/(max_iterations-1) + 1) / max_iterations * max_beta_T

def _boltzmann(energies: npt.NDArray, beta_T: float) -> npt.NDArray:
    """Compute normalised Boltzmann weights for an array of energies.

    Weights are proportional to ``exp(-β_T · E²)`` (squared energy),
    then normalised to sum to one.

    Args:
        energies: 1-D array of energy values (one per shot).
        beta_T: Boltzmann inverse temperature.

    Returns:
        1-D array of non-negative weights summing to 1, same length as
        ``energies``.
    """
    B = np.exp(- beta_T * energies ** 2)
    print(energies[:5], energies[-5:])
    return B / np.sum(B)

def _bias(boltzmanns: npt.NDArray, q: int, samples: list[str]) -> float:
    """Compute the Boltzmann-weighted bias of qubit ``q`` over a sample set.

    Assigns +1 to shots where bit ``q`` is '0' and -1 where it is '1', then
    takes the Boltzmann-weighted average.  A positive bias indicates that the
    ensemble prefers qubit ``q`` in the |0⟩ state.

    Args:
        boltzmanns: Normalised Boltzmann weights, one per shot.
        q: Qubit (bit) index within each bitstring.
        samples: List of measurement bitstrings, one per shot.

    Returns:
        Scalar bias in the range [-1, 1].
    """
    return sum([boltzmanns[i] * (-1 if samples[i][q] == '1' else 1) for i in range(len(samples))])

def _get_biases(
    samples: list[str],
    energies: npt.NDArray,
    beta_T: float
) -> npt.NDArray:
    """Compute per-qubit Boltzmann biases from a measurement ensemble.

    Computes Boltzmann weights from ``energies`` and ``beta_T``, then calls
    ``_bias`` for every qubit.  The output array is reversed (``[::-1]``) to
    match the Qiskit bitstring convention where the rightmost character is
    qubit 0.

    Args:
        samples: List of measurement bitstrings, one per shot.
        energies: 1-D array of energies corresponding to each shot.
        beta_T: Boltzmann inverse temperature.

    Returns:
        1-D array of biases of length ``len(samples[0])``, indexed by qubit
        (qubit 0 is index 0 after the reversal).
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
    """Convert Boltzmann biases into warm-start rotation angles ϕ_i.

    The conversion from bias to angle follows:

        P_i = 0.5 · (1 − η · bias_i)
        P_i = clamp(P_i, eps, 1 − eps)
        ϕ_i = 2 · arcsin(√P_i)

    so that applying ``RY(ϕ_i)`` to |0⟩ prepares a single-qubit state with
    marginal probability ``P_i`` of measuring |1⟩, matching the Boltzmann
    ensemble.

    Args:
        samples: List of measurement bitstrings, one per shot.
        energies: 1-D array of energies, one per shot.
        beta_T: Boltzmann inverse temperature.
        eta: Sign convention (+1 or -1).  Must be in ``{-1, 1}``.
        eps: Probability clamp value; marginals outside ``[eps, 1-eps]`` are
            clipped before the ``arcsin``.

    Returns:
        1-D array of rotation angles of length ``len(samples[0])``.

    Raises:
        Exception: If ``eta`` is not in ``{-1, 1}``.
    """
    if eta not in [-1, 1]:
        raise Exception(f'Eta should be in [-1,1], got {eta}')
    biases = _get_biases(samples, energies, beta_T)
    probabilities = 0.5 * (1 - eta * biases)
    probabilities[probabilities < eps] = eps
    probabilities[probabilities > 1 - eps] = 1 - eps
    
    angles = 2 * np.arcsin(np.sqrt(probabilities))
    return angles

def subsample(samples, energies, alpha=1.0) -> tuple[list[str], npt.NDArray]:
    """Retain only the lowest-energy fraction of a measurement ensemble.

    Sorts samples by ascending energy and returns the top ``alpha`` fraction.
    With ``alpha=1.0`` all samples are returned (no filtering).

    Args:
        samples: List of measurement bitstrings, one per shot.
        energies: 1-D array of energy values corresponding to each shot.
        alpha: Fraction of shots to keep, in the range ``(0, 1]``.
            Only the ``floor(alpha · len(energies))`` lowest-energy shots are
            retained.  Defaults to ``1.0``.

    Returns:
        A tuple ``(sorted_samples, sorted_energies)`` containing only the
        lowest-energy ``alpha`` fraction, both sorted in ascending energy order.
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
    history: list,
    T: Optional[int]=None
):
    """Execute one warm-start QAOA iteration and return updated angles.

    Binds ``angles`` into ``qc``, runs ``shots`` samples via ``sampler``,
    evaluates energies using the Hamiltonian stored in ``data``, optionally
    post-processes counts to enforce timestep constraints, subsamples the
    lowest-energy ``data.alpha`` fraction, computes new Boltzmann biases, and
    converts them to updated warm-start angles via ``_get_angles``.

    The intermediate results ``(samples, energies, mean_energy)`` are appended
    to ``history`` for later analysis.

    Args:
        qc: Parametrised QAOA circuit.  Its parameters must match the layout
            expected by ``assign_parameters(angles)``.
        sampler: Qiskit ``Sampler`` or ``SamplerV2`` primitive used to run the
            circuit.
        shots: Number of measurement shots.
        angles: Current warm-start rotation angles (1-D array of length
            ``num_qubits``), used to bind ``qc``.
        beta_T: Boltzmann inverse temperature for this iteration (see
            ``get_beta_T``).
        data: ``IterativeQAOAData`` instance holding the Hamiltonian,
            ``ising_offset``, ``eta``, ``eps``, and ``alpha``.
        history: Mutable list to which the tuple
            ``[samples, energies, mean_energy]`` is appended.
        T: Number of timesteps for QUBO post-processing (passed to
            ``postprocess``).  Pass ``None`` to skip post-processing.

    Returns:
        Updated warm-start rotation angles (1-D array of length
        ``num_qubits``) derived from the Boltzmann-weighted biases of the
        current ensemble.
    """
    sample_circuit = qc.assign_parameters(angles, inplace=False)
    sampler_job = sampler.run([sample_circuit], shots=shots)
    sampler_result = sampler_job.result()
    counts = sampler_result[0].data.c.get_counts()
    
    if T is not None:
        counts = postprocess(counts, T)
    
    evals = evaluate_sparse_pauli_samples(counts.keys(), data.hamiltonian) + data.ising_offset
    samples, energies = [], []
    for idx, (sample, count) in enumerate(counts.items()):
        samples.extend(count * [sample])
        energies.extend(count * [evals[idx]])
    energies = np.array(energies)
    total_energy = np.mean(energies)
    
    subsamples, subenergies = subsample(samples, energies, data.alpha)
    history.append([samples, energies, total_energy])
    new_angles = _get_angles(
        subsamples, 
        subenergies, 
        beta_T,
        data.eta,
        data.eps
    )
    return new_angles