"""Warm-start iterative HUBO QAOA simulation using a pre-compiled circuit.

Loads a compiled cost circuit from a previous ``circuit_depths.py`` run (all-to-all
coupling, trivial layout) and runs the Boltzmann warm-start iterative QAOA procedure
for a grid of ``(p, rescaling)`` values.

The simulation backend is AerSimulator with the matrix-product-state (MPS) method
(bond dimension 32, single precision, CPU) to keep memory usage tractable for
multi-qubit HUBO circuits.

CLI arguments:
    -f / --filename: Base name of the ``.gfa`` file (without path or extension).
    -n / --shots: Number of measurement shots per warm-start iteration.
    -c / --copy-numbers: Comma-separated copy numbers for each GFA segment.

Hyperparameters (fixed in-script):
    delta_b_fixed: Mixer amplitude ``δ_β = 0.75``.
    delta_g_fixed: Cost amplitude ``δ_γ = 0.30``.
    eta: Bias direction ``η = 1``.
    eps: Probability clipping threshold ``ε = 0.25``.
    max_beta_T: Maximum Boltzmann inverse temperature ``β_T = 0.25``.
    alpha: Sub-sampling fraction ``α = 1.0`` (all samples retained).
    iters: Warm-start iterations per ``(p, rescaling)`` point ``= 5``.

Output pickle schema (saved to
``/lustre/scratch127/qpg/jc59/new_hubo_formulation/nonvariational/<filename>.pkl``):

.. code-block:: python

    {
        'energies': np.ndarray,        # shape (len(ps), len(rescaling))
        'delta_b_fixed': float,
        'delta_g_fixed': float,
        'ps': list[int],
        'rescaling': list[float],
        'samples_dict': dict,          # keyed by (p, rescaling), value is list of
                                       # per-iteration sample lists
    }
"""

import numpy as np
import pickle
import argparse
from itertools import product
from typing import Optional

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as Sampler

from hubo_qaoa.utils.graph_to_hubo_hamiltonian import graph_to_hubo_hamiltonian
from hubo_qaoa.utils.gfa_utils import gfa_file_to_graph
from hubo_qaoa.utils.parameterise_circuit import parameterise_circuit
from hubo_qaoa.utils.lr_qaoa import get_LR_qaoa_circuit
from hubo_qaoa.utils.iterative_qaoa_utils import IterativeQAOAData, iteration, get_beta_T

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
parser.add_argument('-n', '--shots', type=int)
parser.add_argument('-c', '--copy-numbers', help='delimited list input', 
    type=lambda s: [float(item) for item in s.split(',') if len(item)])
args = parser.parse_args()

filename: str = args.filename
shots: int = args.shots

rng = np.random.default_rng()

data_file = '/lustre/scratch127/qpg/jc59/new_hubo_formulation/circuit_depths/results.couplingall.precompute.0.pkl'
with open(data_file, 'rb') as f:
    res = pickle.load(f)
    
cost_circuit = parameterise_circuit(res[filename]['rzz']['circuit'], parameter=Parameter('γ'))
num_qubits: int = cost_circuit.num_qubits    
    
filepath = f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/data/{filename}.gfa'
graph, n, V, T = gfa_file_to_graph(filepath, args.copy_numbers)
hamiltonian, norm = graph_to_hubo_hamiltonian(graph, n, T, lamda=10, constraint_terms=1.0)
hamiltonian = hamiltonian * norm



def warm_start(p: int, delta_b: float, delta_g: float, circ: Optional[QuantumCircuit]=None):
    """Run the Boltzmann warm-start iterative QAOA for a single ``(p, δ_β, δ_γ)`` point.

    Constructs (or reuses) a ``p``-layer linear-ramp QAOA circuit built on top of the
    pre-compiled cost circuit loaded from the ``circuit_depths.py`` output file.
    Runs ``iters = 5`` warm-start iterations, each of which samples the circuit,
    evaluates HUBO energies, and updates the ``Ry`` rotation angles via the Boltzmann
    procedure.

    The pre-compiled cost circuit is loaded once at module level from:
    ``/lustre/scratch127/qpg/jc59/new_hubo_formulation/circuit_depths/
    results.couplingall.precompute.0.pkl``

    and re-parameterised via ``parameterise_circuit`` so that a scalar ``γ`` angle
    can be swept.

    Args:
        p: Number of QAOA layers.
        delta_b: Mixer amplitude ``δ_β`` for the linear-ramp schedule.
        delta_g: Cost amplitude ``δ_γ`` for the linear-ramp schedule.
        circ: Optional existing QAOA circuit to reuse (avoids rebuilding when only
            angles change across warm-start iterations).  ``None`` triggers a fresh
            circuit build.

    Returns:
        A three-tuple ``(energy, samples, circuit)`` where:

        * ``energy`` (``float``) – mean Hamiltonian energy of the final iteration.
        * ``samples`` (``list``) – list of per-iteration sample lists, one entry per
          warm-start iteration.
        * ``circuit`` (``QuantumCircuit``) – the abstract (unbound) QAOA circuit,
          suitable for passing back as ``circ`` in the next call.
    """
    phis = ParameterVector('ϕ', num_qubits)
    fixed_qc, circuit = get_LR_qaoa_circuit(p, delta_b, delta_g, num_qubits, cost_circuit, circ, phis, True)
    history = []
    angles = init_angles
    iters = 5

    for i in range(iters):
        angles = iteration(fixed_qc, sampler, shots, angles, get_beta_T(i, max_beta_T, max_iterations=10), data, history)
        
        
    energy = history[-1][2]
    samples = [history[i][0] for i in range(len(history))]
    logger.info(f'delta_b:{np.round(delta_b, 2)}, delta_g:{np.round(delta_g, 2)}, p:{p}, energy:{np.round(energy, 2)}')
    return energy, samples, circuit
        
        
# delta_b_fixed, delta_g_fixed = 0.45, 0.26
# delta_b_fixed, delta_g_fixed = 0.33, 0.19
delta_b_fixed, delta_g_fixed = 0.75, 0.30

eta = 1
eps = 0.25
max_beta_T = 0.25
alpha = 1.0

probs = 1 / 2 * np.ones((num_qubits,))
#experimental
probs[0] = eps 
thetas = 2 * np.arcsin(np.sqrt(probs))
init_angles = thetas

data = IterativeQAOAData(
    hamiltonian=hamiltonian,
    eta=eta,
    eps=eps,
    alpha=alpha
)

# rescaling = np.logspace(-0.5, 0.2, 8, base=10)
rescaling= [1,]
ps = [1,3,5]

energies = np.zeros((len(ps), len(rescaling)))
samples_dict = {}

circuit = None
for i, j in product(range(len(ps)), range(len(rescaling))):
    if j == 0:
        circuit = None
    e, samples, circuit = warm_start(ps[i], delta_b_fixed * rescaling[j], delta_g_fixed * rescaling[j], circuit)
    energies[i, j] = e
    samples_dict[(ps[i], rescaling[j])] = samples
    
    
to_save=dict(energies=energies, delta_b_fixed=delta_b_fixed, delta_g_fixed=delta_g_fixed, ps=ps, rescaling=rescaling, samples_dict=samples_dict)    
with open(f'/lustre/scratch127/qpg/jc59/new_hubo_formulation/nonvariational/nonvariational.{filename}.db{delta_b_fixed}.dg{delta_g_fixed}.ps{ps[-1]}.shots{shots}.betaT{max_beta_T}.eps{eps}.alpha{alpha}.pkl', 'wb') as f:
    pickle.dump(to_save, f)