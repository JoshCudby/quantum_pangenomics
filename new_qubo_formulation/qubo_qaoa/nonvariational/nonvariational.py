"""Iterative warm-start QAOA with fixed linear-ramp parameters.

This script runs the full Iter-QAOA procedure on a pre-built QUBO Hamiltonian
using the non-variational linear-ramp (LR) parameter schedule.  Parameters
are **not** optimised variationally; instead, fixed amplitudes ``delta_b`` and
``delta_g`` define the schedule via:

    β_j = Δβ · (1 − (j − 0.5) / p)
    γ_j = Δγ · (j − 0.5) / p,   j = 1, …, p

After fixing the circuit, ``max_iterations`` (= 10) Boltzmann warm-start
iterations are performed, each refining the single-qubit initialisation angles
``ϕ_i`` based on the weighted biases of the previous sample set.  The inverse
temperature β_T used for Boltzmann weighting follows a quadratic schedule from
``max_beta_T / max_iterations`` up to ``max_beta_T`` (see ``get_beta_T``).

Hyperparameters used:

* ``delta_b = 0.63`` — mixer schedule amplitude.
* ``delta_g = 0.16`` — cost schedule amplitude.
* ``max_beta_T = 0.15`` — maximum Boltzmann inverse temperature.
* ``eta = 1`` — bias sign convention (warm-start towards low-energy |0⟩ states).
* ``eps = 0.05`` — probability clamp for ``arcsin`` stability.
* ``alpha = 1.0`` — no subsampling (all shots used for bias computation).
* ``max_iterations = 10`` — number of warm-start refinement steps.

The Aer simulator uses matrix-product-state (MPS) simulation with bond
dimension 32 for memory-efficient simulation of large qubit counts.

CLI Arguments:
    -f / --filename: Stem of the input ``.gfa`` file (the script appends
        ``.gfa.pkl`` and looks in a fixed Lustre path).
    -N / --nodes: Number of nodes N in the pangenome graph.  Used to set the
        initial warm-start angle ``ϕ = 2·arcsin(√(1/(2N)))``.
    -T / --time: (Unused placeholder; reserved for future timestep argument.)
    -n / --shots: Number of measurement shots per warm-start iteration.

Output:
    A pickle file at
    ``/lustre/.../nonvariational/nonvariational.<filename>.db<Δβ>.dg<Δγ>.shots<n>.betaT<β_T>.eps<ε>.alpha<α>.pkl``
    containing a dict with keys ``energies``, ``delta_b_fixed``,
    ``delta_g_fixed``, ``ps``, ``rescaling``, ``samples_dict``.
"""

import numpy as np
import numpy.typing as npt
import pickle
import argparse
from itertools import product
from typing import Optional

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as Sampler

from qubo_qaoa.utils.lr_qaoa import get_LR_qaoa_circuit
from qubo_qaoa.utils.iterative_qaoa_utils import IterativeQAOAData, iteration, get_beta_T

from qiskit_qaoa.utils.hamiltonian_utils import get_Q_and_hamiltonian
from qiskit_qaoa.utils.logging import get_logger

logger = get_logger(__name__)

backend_options = dict(
    method='matrix_product_state',
    matrix_product_state_max_bond_dimension='32', 
    device='CPU',
    precision='single',
    basis_gates = ['rx', 'ry', 'rz', 'cx']
)
backend = AerSimulator(**backend_options)
sampler = Sampler.from_backend(backend)

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', type=str)
parser.add_argument('-N', '--nodes', type=int)
parser.add_argument('-T', '--time', type=int)
parser.add_argument('-n', '--shots', type=int)
args = parser.parse_args()

filename: str = args.filename
N: int = args.nodes
shots: int = args.shots

rng = np.random.default_rng()

data_file = f'/lustre/scratch127/qpg/jc59/new_qubo_formulation/oriented/qubo_data/qubo_data_{filename}.gfa.pkl'

_, hamiltonian, _, ising_offset = get_Q_and_hamiltonian(data_file)
num_qubits: int = hamiltonian.num_qubits


def warm_start(
    p: int,
    delta_b: float,
    delta_g: float,
    circ: Optional[QuantumCircuit]=None
) -> tuple[float, list[list[str]], QuantumCircuit]:
    """Run iterative Boltzmann warm-start QAOA for a single (p, Δβ, Δγ) point.

    Builds a p-layer LR-QAOA circuit with warm-start angle parameters ``ϕ``,
    fixes the circuit parameters to the linear-ramp schedule determined by
    ``delta_b`` and ``delta_g``, then performs ``iters=10`` warm-start
    iterations.  Each iteration:

    1. Samples ``shots`` bitstrings from the current warm-start circuit.
    2. Evaluates energies via the module-level ``hamiltonian`` and
       ``ising_offset``.
    3. Updates ``ϕ`` via Boltzmann-weighted biases (see ``iteration`` in
       ``iterative_qaoa_utils``).

    The Boltzmann inverse temperature ``β_T`` follows the quadratic schedule
    ``get_beta_T(i, max_beta_T)`` across iterations.

    Args:
        p: Number of QAOA layers.
        delta_b: Mixer amplitude Δβ for the LR schedule.
        delta_g: Cost amplitude Δγ for the LR schedule.
        circ: Optional pre-built parametrised circuit template.  If ``None``,
            a new circuit is constructed (which involves Aer transpilation).
            Reusing ``circ`` across calls with the same ``p`` avoids redundant
            transpilation.

    Returns:
        A tuple ``(energy, samples, circuit)`` where:

        * ``energy`` is the mean QUBO energy over the final iteration's shots.
        * ``samples`` is a list (length ``iters``) of per-iteration bitstring
          lists.
        * ``circuit`` is the (possibly freshly built) parametrised circuit
          template for reuse.
    """
    phis = ParameterVector('ϕ', num_qubits)
    fixed_qc, circuit = get_LR_qaoa_circuit(
        p, delta_b, delta_g, num_qubits,
        hamiltonian, circ, phis=phis, measure=True
    )
    
    history = []
    angles = init_angles
    iters = 10
    
    for i in range(iters):
        angles = iteration(fixed_qc, sampler, shots, angles, get_beta_T(i, max_beta_T), data, history, T=None)
        
    energy = history[-1][2]
    samples = [history[i][0] for i in range(len(history))]
    logger.info(f'delta_b:{np.round(delta_b, 2)}, delta_g:{np.round(delta_g, 2)}, p:{p}, energy:{np.round(energy, 2)}')
    return energy, samples, circuit

delta_b_fixed = 0.63
delta_g_fixed = 0.16
        
eta = 1
eps = 0.05
max_beta_T =  0.15
alpha = 1.0

data = IterativeQAOAData(
    hamiltonian=hamiltonian,
    ising_offset=ising_offset,
    eta=eta,
    eps=eps,
    alpha=alpha
)

# init_angles = np.pi/2 * np.ones((num_qubits,))
prob = 1 / (2 * N)
theta = 2 * np.arcsin(np.sqrt(prob))
init_angles: npt.NDArray = theta * np.ones((num_qubits,))


# rescaling = np.logspace(-0.5, 0.2, 8, base=10)
# ps = sorted(set([int(x) for x in np.logspace(0, 1.5, 3, base=10)]))
rescaling = np.array([1,])
ps = [1]


# MAIN
energies = np.zeros((len(ps), len(rescaling)))
samples_dict: dict[tuple[int, float], list[list[str]]] = {}

circuit = None
for i, j in product(range(len(ps)), range(len(rescaling))):
    if j == 0:
        circuit = None
    e, samples, circuit = warm_start(ps[i], delta_b_fixed * rescaling[j], delta_g_fixed * rescaling[j], circuit)
    energies[i, j] = e
    samples_dict[(ps[i], np.round(rescaling[j],3))] = samples
    
to_save=dict(energies=energies,  delta_b_fixed=delta_b_fixed, delta_g_fixed=delta_g_fixed, ps=ps, rescaling=rescaling, samples_dict=samples_dict)    
with open(f'/lustre/scratch127/qpg/jc59/new_qubo_formulation/oriented/nonvariational/nonvariational.{filename}.db{delta_b_fixed}.dg{delta_g_fixed}.shots{shots}.betaT{max_beta_T}.eps{eps}.alpha{alpha}.pkl', 'wb') as f:
    pickle.dump(to_save, f)