"""Parameter landscape sweep for HUBO QAOA.

Evaluates the HUBO QAOA energy and optimal-state overlap ``p_opt`` over a dense grid
of ``(δ_β, δ_γ, p)`` values using statevector simulation (GPU, single precision).
This is used to identify good linear-ramp QAOA hyperparameters before committing to
more expensive shot-based or hardware runs.

The ``--normalise`` flag controls which Hamiltonian is used to evaluate energies in
the statevector inner product:

* Without ``--normalise`` (default): energies are evaluated against the **full**
  unnormalised Hamiltonian ``H = λ · H_constraint + H_objective``, giving energies
  in physical units.
* With ``--normalise``: energies are evaluated against the **normalised** Hamiltonian
  (coefficients in ``[−1, 1]``) as used during circuit compilation.  This allows
  direct comparison with compiled-Hamiltonian energy landscapes when the normalisation
  constant changes between runs.

CLI arguments:
    -f / --filename: Base name of the ``.gfa`` file (without path or extension).
    -c / --copy-numbers: Comma-separated copy numbers for each GFA segment.
    --normalise: If set, evaluate against the normalised (compiled) Hamiltonian.

Output pickle schema (saved to
``/lustre/scratch127/qpg/jc59/new_hubo_formulation/nonvariational/param_exploration/
LR_unequal.<filename>.db<delta_bs[-1]>.dg<delta_gs[-1]>.p<ps[-1]>.pkl``):

.. code-block:: python

    {
        'delta_bs': np.ndarray,   # shape (41,), linspace(0, 1, 41)
        'delta_gs': np.ndarray,   # shape (41,), linspace(0, 2, 41)
        'ps': list[int],          # [1, 2, 3, 4, 5]
        'energies': np.ndarray,   # shape (len(ps), 41, 41)
        'p_opts': np.ndarray,     # shape (len(ps), 41, 41)
    }
"""

import numpy as np
from typing import Optional
import pickle
import argparse
from itertools import product

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as Sampler


from hubo_qaoa.utils.graph_to_hubo_hamiltonian import graph_to_hubo_hamiltonian
from hubo_qaoa.utils.gfa_utils import gfa_file_to_graph
from hubo_qaoa.utils.parameterise_circuit import parameterise_circuit
from hubo_qaoa.utils.lr_qaoa import get_LR_qaoa_circuit
from qiskit_qaoa.utils.string_utils import evaluate_sparse_pauli_samples_all
from qiskit_qaoa.utils.logging import get_logger


logger = get_logger(__name__)


backend_options = dict(
    # method='matrix_product_state',
    # matrix_product_state_max_bond_dimension='20', 
    method='statevector',
    device='GPU',
    precision='single',
    basis_gates = ['rx', 'ry', 'rz', 'cx']
)
backend = AerSimulator(**backend_options)
sampler = Sampler.from_backend(backend)

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', type=str)
parser.add_argument('-c', '--copy-numbers', help='delimited list input', 
    type=lambda s: [float(item) for item in s.split(',') if len(item)])
parser.add_argument('--normalise', action='store_true', default=False)
args = parser.parse_args()


filename: str = args.filename

filepath = f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/data/{filename}.gfa'
graph, n, V, T = gfa_file_to_graph(filepath, args.copy_numbers)
normalised_hamiltonian, norm = graph_to_hubo_hamiltonian(graph, n, T, lamda=10, constraint_terms=1.0)
hamiltonian = normalised_hamiltonian * norm

circuit_hamiltonian = normalised_hamiltonian if args.normalise else hamiltonian

data_file = '/lustre/scratch127/qpg/jc59/new_hubo_formulation/circuit_depths/results.couplingall.precompute.0.pkl'
with open(data_file, 'rb') as f:
    res = pickle.load(f)
cost_circuit = parameterise_circuit(res[filename]['rzz']['circuit'], parameter=Parameter('γ'))


num_qubits: int = cost_circuit.num_qubits    
    
evals = evaluate_sparse_pauli_samples_all(hamiltonian)
opt_evals = np.nonzero(evals < 1e-5)
print(f'Opt evals: {opt_evals}')


def get_energy_and_p_opt(qc) -> tuple[float, float]:
    """Evaluate the expected energy and optimal-state overlap for a statevector circuit.

    Runs the circuit with ``shots=1`` on the statevector backend (which returns an
    exact statevector regardless of shot count) and computes:

    * ``energy = ∑_x |⟨x|ψ⟩|² · E(x)`` – expected Hamiltonian energy under the
      full unnormalised Hamiltonian eigenvalues.
    * ``p_opt = ∑_{x: E(x)<10⁻⁵} |⟨x|ψ⟩|²`` – total probability weight on
      computational basis states with near-zero energy (optimal solutions).

    Args:
        qc: A fully bound (no free parameters) ``QuantumCircuit`` ending with a
            ``save_statevector`` instruction.

    Returns:
        A two-tuple ``(energy, p_opt)`` where both are real scalars.
    """
    job = backend.run([qc],shots=1)
    result = job.result()
    data = result.results[0].data

    sv = np.asarray(data.statevector)
    energy = np.sum(np.abs(sv) ** 2 * evals)
    p_opt = np.sum(np.abs(sv[opt_evals]) ** 2)
    return energy,p_opt



def LR_QAOA(p: int, delta_b: float, delta_g: float, circ: Optional[QuantumCircuit]):
    """Evaluate energy and ``p_opt`` for a single LR-QAOA ``(p, δ_β, δ_γ)`` point.

    Constructs (or reuses) a ``p``-layer linear-ramp QAOA circuit without
    warm-start angles (standard Hadamard initial state, standard Rx mixer), binds
    the LR-schedule parameters, and evaluates the statevector via
    ``get_energy_and_p_opt``.

    Unlike ``warm_start`` in ``nonvariational.py``, no Boltzmann iteration is
    performed here; this is a single-shot statevector evaluation for landscape
    mapping.

    Args:
        p: Number of QAOA layers.
        delta_b: Mixer amplitude ``δ_β``.
        delta_g: Cost amplitude ``δ_γ``.
        circ: Optional existing abstract QAOA circuit to reuse.  ``None`` triggers
            a fresh build.

    Returns:
        A three-tuple ``(energy, p_opt, circuit)`` where:

        * ``energy`` (``float``) – expected Hamiltonian energy.
        * ``p_opt`` (``float``) – total probability on near-zero-energy states.
        * ``circuit`` (``QuantumCircuit``) – abstract (unbound) circuit for reuse.
    """
    fixed_qc, circuit = get_LR_qaoa_circuit(p, delta_b, delta_g, num_qubits, cost_circuit, circ, None, None)

    energy, p_opt = get_energy_and_p_opt(fixed_qc)
        
    logger.info(f'delta_b:{np.round(delta_b, 2)}, delta_g:{np.round(delta_g, 2)}, p:{p}, energy:{np.round(energy, 2)}, p_opt: {np.round(p_opt, 4)}')
    return energy, p_opt, circuit
    
eps = 1e-2
  

# delta_bs = np.logspace(-1.0, 0.5, 21, base=10)
# delta_gs = np.logspace(-1.0, 0.5, 21, base=10)
delta_bs = np.linspace(0.0, 1, 41)
delta_gs = np.linspace(0.0, 2, 41)

# ps = [int(x) for x in np.logspace(0, 2.5, 6, base=10)]
ps = [1,2,3,4,5]

energies = np.zeros((len(ps), len(delta_bs), len(delta_gs)))
p_opts = np.zeros((len(ps), len(delta_bs), len(delta_gs)))

circuit = None
for i, j, k in product(range(len(ps)), range(len(delta_bs)), range(len(delta_gs))):
    if j == 0 and k == 0:
        circuit = None
    e, p_opt, circuit = LR_QAOA(ps[i], delta_bs[j], delta_gs[k], circuit)
    energies[i, j, k] = e
    p_opts[i, j, k] = p_opt

to_save_name = f'/lustre/scratch127/qpg/jc59/new_hubo_formulation/nonvariational/param_exploration/LR_unequal.{filename}.db{np.round(delta_bs[-1],2)}.dg{np.round(delta_gs[-1],2)}.p{ps[-1]}.pkl'
ret = dict(delta_bs=delta_bs, delta_gs=delta_gs, ps=ps, energies=energies, p_opts=p_opts)
    
with open(to_save_name, 'wb') as f:
    pickle.dump(ret, f)

# delta_bs = np.linspace(0, 1, 200)
# ps = range(1, 1002, 10)

# energies = np.zeros((len(ps), len(delta_bs)))
# circuit = None
# for i, j in product(range(len(ps)), range(len(delta_bs))):
#     if j == 0:
#         circuit = None
#     e, circuit = LR_QAOA(ps[i], delta_bs[j], delta_bs[j], circuit)
#     energies[i, j] = e

# to_save_name = f'/lustre/scratch127/qpg/jc59/new_hubo_formulation/nonvariational/param_exploration/LR_equal.{filename}.d{np.round(delta_bs[-1],2)}.p{ps[-1]}.normalise{args.normalise}.pkl'
# ret = dict(deltas=delta_bs, ps=ps, energies=energies)
    
# with open(to_save_name, 'wb') as f:
#     pickle.dump(ret, f)