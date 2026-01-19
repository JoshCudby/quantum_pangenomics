
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
    '''Class for storing global parameters of an Iter-QAOA Run.'''
    hamiltonian: SparsePauliOp
    ising_offset: float
    eta: int
    eps: float
    alpha: float
    

def get_beta_T(i: int, max_beta_T: float):
    # A quadratic ramp from 0.1 to 1 over 10 iterations, scaled by max_beta_T
    return ((i ** 2)/9 + 1) / 10 * max_beta_T

def _boltzmann(energies: npt.NDArray, beta_T: float) -> npt.NDArray:
    B = np.exp(- beta_T * energies ** 2)
    return B / np.sum(B)

def _bias(boltzmanns: npt.NDArray, q: int, samples: list[str]) -> float:
    return sum([boltzmanns[i] * (-1 if samples[i][q] == '1' else 1) for i in range(len(samples))])

def _get_biases(
    samples: list[str],
    energies: npt.NDArray, 
    beta_T: float
) -> npt.NDArray:
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
    if eta not in [-1, 1]:
        raise Exception(f'Eta should be in [-1,1], got {eta}')
    biases = _get_biases(samples, energies, beta_T)
    probabilities = 0.5 * (1 - eta * biases)
    probabilities[probabilities < eps] = eps
    probabilities[probabilities > 1 - eps] = 1 - eps
    
    angles = 2 * np.arcsin(np.sqrt(probabilities))
    return angles

def subsample(samples, energies, alpha=1.0) -> tuple[list[str], npt.NDArray]:
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
    sample_circuit = qc.assign_parameters(angles, inplace=False)
    sampler_job = sampler.run([sample_circuit], shots=shots)
    sampler_result = sampler_job.result()
    counts = sampler_result[0].data.c.get_counts()
    
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