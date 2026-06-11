"""Parameter sweep over (Δβ, Δγ, p) to find optimal LR-QAOA settings.

This script performs a dense grid search over the linear-ramp QAOA parameter
space to identify the amplitude pair (Δβ*, Δγ*) and circuit depth p* that
minimise the mean QUBO energy (or maximise the probability of sampling an
optimal solution, ``p_opt``).

The sweep covers:

* ``delta_bs``: 41 evenly-spaced values in [0, 1] (mixer amplitudes Δβ).
* ``delta_gs``: 41 evenly-spaced values in [0, 1] (cost amplitudes Δγ).
* ``ps``: p ∈ {1, 2, 3, 4, 5} (number of QAOA layers).

For each triple ``(p, Δβ, Δγ)`` the script either runs the circuit with
``--measure`` (MPS sampling) or evaluates the exact statevector (GPU
statevector simulation), reporting both the mean energy and the probability of
sampling an exact optimum (``p_opt``).

CLI Arguments:
    -f / --filename: Stem of the input ``.gfa`` file.
    --measure: Flag.  If set, uses MPS sampling (CPU); otherwise uses exact
        GPU statevector simulation.
    -n / --shots: Number of shots per circuit evaluation (default 4000,
        relevant only when ``--measure`` is set).

Output:
    A pickle file at
    ``/lustre/.../param_exploration/LR_unequal.<filename>.db<Δβ_max>.dg<Δγ_max>.p<p_max>.pkl``
    containing a dict with keys:

    * ``delta_bs``: Array of Δβ values swept.
    * ``delta_gs``: Array of Δγ values swept.
    * ``ps``: List of p values swept.
    * ``energies``: Array of shape ``(len(ps), len(delta_bs), len(delta_gs))``
      with mean QUBO energy for each triple.
    * ``p_opts``: Array of the same shape with the probability of sampling an
      optimal bitstring.
"""

import numpy as np
import pickle
import argparse
from itertools import product

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as Sampler

from qubo_qaoa.utils.lr_qaoa import get_LR_qaoa_circuit
from qubo_qaoa.utils.str_utils import genbin

from qiskit_qaoa.utils.hamiltonian_utils import get_normalised_Q_and_hamiltonian
from qiskit_qaoa.utils.string_utils import evaluate_sparse_pauli_samples_all, evaluate_sparse_pauli_samples
from qiskit_qaoa.utils.logging import get_logger

logger = get_logger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', type=str)
parser.add_argument('--measure', action='store_true', default=False)
parser.add_argument('-n', '--shots', default=4000, type=int)
args = parser.parse_args()
logger.info(args)
filename: str = args.filename
measure = args.measure

if measure:
    backend_options = dict(
        method='matrix_product_state',
        matrix_product_state_max_bond_dimension='32', 
        # method='statevector',
        device='CPU',
        precision='single',
        basis_gates = ['rx', 'ry', 'rz', 'cx']
    )
else:
    backend_options = dict(
        method='statevector',
        device='GPU',
        precision='single',
        basis_gates = ['rx', 'ry', 'rz', 'cx']
    ) 
# fake_fez = FakeFez()
backend = AerSimulator(**backend_options)
sampler = Sampler.from_backend(backend)



rng = np.random.default_rng()

data_file = f'/lustre/scratch127/qpg/jc59/new_qubo_formulation/oriented/qubo_data/qubo_data_{filename}.gfa.pkl'

_, hamiltonian, _, ising_offset, _, ham_norm = get_normalised_Q_and_hamiltonian(data_file)
hamiltonian = hamiltonian * ham_norm
ising_offset = ising_offset * ham_norm
num_qubits: int = hamiltonian.num_qubits

if not measure:
    evals = evaluate_sparse_pauli_samples_all(hamiltonian) + ising_offset
    opt_evals = np.nonzero(evals < 1e-5)
    print(f'Opt evals: {opt_evals}')


def get_energy_and_p_opt(qc):
    """Sample a bound circuit and return mean energy and optimal-solution probability.

    Runs ``args.shots`` shots via the module-level ``sampler``, evaluates each
    unique bitstring against the module-level ``hamiltonian`` (plus
    ``ising_offset``), then computes the mean energy and the fraction of shots
    that hit an exact optimum (energy < 1e-5).

    Args:
        qc: A fully bound (parameter-free) ``QuantumCircuit`` ready to sample.

    Returns:
        A tuple ``(energy, p_opt)`` where ``energy`` is the mean QUBO energy
        across all shots and ``p_opt`` is the fraction of shots with energy
        below ``1e-5``.
    """
    job = sampler.run([qc], shots=args.shots)
    sampler_result = job.result()
    counts = sampler_result[0].data.meas.get_counts()
    
    evals = evaluate_sparse_pauli_samples(counts.keys(), hamiltonian) + ising_offset
    samples, energies = [], []
    for idx, (sample, count) in enumerate(counts.items()):
        samples.extend(count * [sample])
        energies.extend(count * [evals[idx]])
        
    energies = np.array(energies)
    energy = np.mean(energies)
    p_opt = np.flatnonzero(energies < 1e-5).shape[0] / args.shots
    return energy, p_opt


def get_energy_and_p_opt_sv(qc):
    """Evaluate mean energy and optimal probability from a statevector simulation.

    Runs the circuit as a statevector (single-shot, GPU backend) and computes
    the exact expectation value and exact optimal-solution probability using
    the pre-computed ``evals`` array.

    Args:
        qc: A fully bound ``QuantumCircuit`` with a ``save_statevector``
            instruction appended (produced by ``get_LR_qaoa_circuit`` when
            ``measure=False``).

    Returns:
        A tuple ``(energy, p_opt)`` where ``energy`` is ``∑|ψ_i|² · evals_i``
        and ``p_opt`` is ``∑_{i ∈ opt} |ψ_i|²``.
    """
    result = backend.run([qc],shots=1).result()
    data = result.results[0].data
    sv = np.asarray(data.statevector)
    energy = np.sum(np.abs(sv) ** 2 * evals)
    p_opt = np.sum(np.abs(sv[opt_evals]) ** 2)
    return energy, p_opt


def LR_QAOA(p, delta_b, delta_g, circ):
    """Evaluate the LR-QAOA energy and p_opt for a single (p, Δβ, Δγ) point.

    Builds (or reuses) the parametrised circuit template ``circ``, binds the
    linear-ramp parameters, and dispatches to either ``get_energy_and_p_opt``
    (sampling) or ``get_energy_and_p_opt_sv`` (statevector) depending on the
    ``--measure`` flag.

    Args:
        p: Number of QAOA layers.
        delta_b: Mixer amplitude Δβ.
        delta_g: Cost amplitude Δγ.
        circ: Pre-built parametrised circuit template for reuse, or ``None``
            to trigger fresh construction.

    Returns:
        A tuple ``(energy, p_opt, circuit)`` where ``energy`` is the mean
        QUBO energy, ``p_opt`` is the probability of sampling an optimum, and
        ``circuit`` is the (possibly freshly built) template for the next call.
    """
    fixed_qc, circuit = get_LR_qaoa_circuit(
        p, delta_b, delta_g, num_qubits,
        hamiltonian, circ, phis=None, measure=measure
    )
    
    if measure:
        energy, p_opt = get_energy_and_p_opt(fixed_qc)
    else:
        energy, p_opt = get_energy_and_p_opt_sv(fixed_qc)
        
    logger.info(f'delta_b:{np.round(delta_b, 2)}, delta_g:{np.round(delta_g, 2)}, p:{p}, energy:{np.round(energy, 2)}, p_opt: {np.round(p_opt, 4)}')
    return energy, p_opt, circuit
        
        

eps = 1e-2
# delta_bs = np.logspace(-0.5, 0.5, 41, base=10)
# delta_gs = np.logspace(-1.5, -0.5, 41, base=10)
delta_bs = np.linspace(0.0, 1, 41) 
delta_gs = np.linspace(0.0, 1, 41)


# ps = sorted(set([int(x) for x in np.logspace(0, 2, 5, base=10)]))
ps = range(1, 6)

energies = np.zeros((len(ps), len(delta_bs), len(delta_gs)))
p_opts = np.zeros((len(ps), len(delta_bs), len(delta_gs)))
circuit = None
for i, j, k in product(range(len(ps)), range(len(delta_bs)), range(len(delta_gs))):
    if j == 0 and k == 0:
        circuit = None
    e, p_opt, circuit = LR_QAOA(ps[i], delta_bs[j], delta_gs[k], circuit)
    energies[i, j, k] = e
    p_opts[i, j, k] = p_opt

to_save_name = f'/lustre/scratch127/qpg/jc59/new_qubo_formulation/oriented/param_exploration/LR_unequal.{filename}.db{np.round(delta_bs[-1],2)}.dg{np.round(delta_gs[-1],2)}.p{ps[-1]}.pkl'
ret = dict(delta_bs=delta_bs, delta_gs=delta_gs, ps=ps, energies=energies, p_opts=p_opts)
    
    
with open(to_save_name, 'wb') as f:
    pickle.dump(ret, f)
