"""HUBO QAOA parameter optimisation targeting real IBM quantum hardware.

Loads a compiled HUBO circuit produced by ``hubo_circuit_compilation.py``,
wraps it in a full QAOA ansatz (p repetitions), transpiles to the least-busy
IBM backend via QiskitRuntimeService, then optimises the beta/gamma parameters
using ``scipy.optimize.minimize``.

Objective function:
    The CVaR (Conditional Value-at-Risk) of the bitstring energies evaluated
    against the remapped full Hamiltonian, controlled by the ``--alpha``
    parameter.  CVaR with alpha < 1 focuses the objective on the lowest-energy
    tail of the sample distribution.

Parameter initialisation:
    - ``ramp``:   linearly spaced schedule based on QAOA annealing heuristic.
    - ``random``: uniform random in [0, 1].
    - ``warm``:   not yet implemented.

Serialisation:
    The optimisation result is pickled to::

        <basepath>/optimisation.<f>.extra<e>.times<t>.four<frac4>.six<frac6>
            .method<M>.cvar<a>.p<p>.shots<n>.init<init>.d<d>.pkl

    The pickle contains::

        {
            'result':                   scipy.optimize.OptimizeResult,
            'history':                  list of (sampling_time, energy, params,
                                                 counts, post_process_time),
            'remapped_full_hamiltonian': SparsePauliOp,
            't_qaoa_circ':              QuantumCircuit,
            'compiled_hamiltonian':     SparsePauliOp,
            'layout':                   Layout,
        }

CLI arguments:
    -f / --filename:       GFA file stem (used to locate the compilation
                           pickle).
    -p / --reps:           Number of QAOA layers p (default 4).
    -d / --swap-depth:     SWAP-layer budget index to use from the compilation
                           pickle (default 0).
    -m / --memory:         GPU memory limit in MB (default 16000).
    -M / --method:         scipy optimiser name (e.g. ``COBYLA``, ``L-BFGS-B``).
    -n / --shots:          Shots per circuit evaluation (default 1000).
    --init:                Parameter initialisation strategy: ``ramp``,
                           ``random``, or ``warm``.
    -e / --extra:          Extra SWAP layers used during compilation (default 1).
    --fraction-four:       Fraction of 4-body terms retained at compilation.
    --fraction-six:        Fraction of 6-body terms retained at compilation.
    --times-to-keep:       Timestep-transition indices used at compilation.
    -N / --nodes:          Number of graph nodes (used to derive n).
    -T / --time:           Number of timesteps T.
    -a / --alpha:          CVaR tail fraction in (0, 1] (default 0.25).
"""
import numpy as np
import networkx as nx
from itertools import combinations
from time import time
import pickle
import argparse
from scipy.optimize import minimize, OptimizeResult

from qiskit import QuantumCircuit, generate_preset_pass_manager
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.circuit import Parameter
from qiskit.transpiler import Layout

from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2 as Sampler
 
from qopt_best_practices.transpilation.qaoa_construction_pass import QAOAConstructionPass

# from qiskit_qaoa.utils.qaoa_circuit_utils import get_mixer_operator, state_prep
from qiskit_qaoa.utils.transpiler_passes import ExtendedSwapStrategy
from qiskit_qaoa.utils.pass_managers import get_hubo_pass_manager
from qiskit_qaoa.utils.string_utils import evaluate_sparse_pauli_samples
from qiskit_qaoa.utils.logging import get_logger


def print_circuit_info(qc, circuit_name):
    """Log the 2-qubit gate count and 2-qubit gate depth of a circuit.

    Args:
        qc: The quantum circuit to summarise.
        circuit_name: A human-readable label included in the log message.
    """
    logger.info(f'{circuit_name} has {qc.count_ops().get("cz", 0) + qc.count_ops().get("rzz", 0) + qc.count_ops().get("cx", 0)} 2Q gates \
    and {qc.depth(lambda instr: len(instr.qubits) > 1)} 2Q depth')
        

logger = get_logger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename')
parser.add_argument('-p', '--reps', type=int, default=4)
parser.add_argument('-d', '--swap-depth', type=int, default=0)
parser.add_argument('-m', '--memory', type=int, default=16000)
parser.add_argument('-M', '--method', type=str)
parser.add_argument('-n', '--shots', type=int, default=1000)
parser.add_argument('--init', choices=['ramp', 'random', 'warm'], default='ramp')
parser.add_argument('-e', '--extra', type=int, default=1)
parser.add_argument('--fraction-four', type=float)
parser.add_argument('--fraction-six', type=float)
parser.add_argument('--times-to-keep', help='delimited list input', 
    type=lambda s: tuple([int(item) for item in s.split(',') if len(item)]))
parser.add_argument('-N', '--nodes', type=int)
parser.add_argument('-T', '--time', type=int)
parser.add_argument('-a', '--alpha', type=float, default=0.25)


args = parser.parse_args()

logger.info(args)

filename: str = args.filename
p: int = args.reps
shots: int = args.shots
init_type: str = args.init
swap_depth: int = args.swap_depth
N: int = args.nodes
T: int = args.time
n = int(np.ceil(np.log2(2*N+1)))

seed = 1
rng = np.random.default_rng()

basis_gates=["sx", "x", "rz", "rzz", "cz", "id", "swap", "cx", "h"]


basepath = '/lustre/scratch127/qpg/jc59/hubo_hardware/'
filename = 'compilation.{}.extra{}.times{}.four{}.six{}'.format(
    args.filename,
    args.extra,
    ''.join([str(t) for t in args.times_to_keep]),
    args.fraction_four,
    args.fraction_six
)
results_file = basepath + filename + '.pkl'

with open(results_file, 'rb') as f:
    data = pickle.load(f)
    compiled_hamiltonian: SparsePauliOp = data['compiled_hamiltonian']
    full_hamiltonian: SparsePauliOp = data['full_hamiltonian']
    layout: Layout = data[swap_depth]

num_qubits: int = full_hamiltonian.num_qubits if full_hamiltonian.num_qubits is not None else max(layout.get_physical_bits().keys())

rows, cols = 0, 0
while 2 * (rows + cols + rows * cols) < num_qubits:
    if rows < cols:
        rows += 1
    else:
        cols += 1
logger.info(f'Min size to support virtual qubits: {(rows, cols)}, ')

extended_swap_strat = ExtendedSwapStrategy.from_heavy_hex(rows, cols)
num_physical_qubits = extended_swap_strat._num_vertices
coupling_map = extended_swap_strat._coupling_map

service = QiskitRuntimeService(name='eu_test_instance')
backend = service.least_busy(min_num_qubits=num_qubits, operational=True, simulator=False) 
logger.info(f'Backend: {backend}')
logger.info(f'Num qubits in backend: {backend.configuration().to_dict()["n_qubits"]}')

donor_qc = QuantumCircuit(num_physical_qubits)
remapped_full_hamiltonian = full_hamiltonian.apply_layout([layout.get_virtual_bits()[donor_qc.qubits[i]] for i in range(num_qubits)], num_physical_qubits)


logger.info(f'Physical qubits: {num_physical_qubits}')

coupling_map_edge = list(coupling_map)
physical_qubits = list(coupling_map.physical_qubits)
dual_coupling_map = nx.Graph()

for qubit in physical_qubits:
    edges = [edge for edge in coupling_map_edge if edge[0]==qubit]
    for edge1, edge2 in combinations(edges, 2):
        dual_coupling_map.add_edge(tuple(sorted(edge1)), tuple(sorted(edge2)))
edge_colouring = nx.greedy_color(dual_coupling_map, interchange=True)


pm = get_hubo_pass_manager(extended_swap_strat, swap_depth, args.extra)

cost_qc = QuantumCircuit(num_physical_qubits)
cost_qc.append(PauliEvolutionGate(compiled_hamiltonian, time=Parameter("c")), [layout.get_virtual_bits()[donor_qc.qubits[i]] for i in range(num_qubits)])
tcost_qc = pm.run(cost_qc)

print_circuit_info(tcost_qc, 'Transpiled cost hamiltonian circuit')
print(tcost_qc.count_ops())
logger.info(f'Cost hamiltonian circuit has {tcost_qc.num_qubits} qubits')


# TODO: instead of using construction pass, use p different cost hamiltonians with different mappings
# Can't do different mappings since the qubit locations are now set.. but could do the next N layers of SWAP strat
# Which would allow for a different subset of interactions to be used

if not 2*N+1 == 2**(int(np.log2(2*N+1))):
    # sp = state_prep(N,T)
    # mixer = get_mixer_operator(N,T)
    # logger.info('Using Grover mixer and state prep')
    sp = None
    mixer = None
    logger.info('Using X mixer and Hadamard state prep')
else:
    sp = None
    mixer = None
    logger.info('Using X mixer and Hadamard state prep')
    
    
construction_pass = QAOAConstructionPass(p, init_state=sp, mixer_layer=mixer)
qaoa_circ = dag_to_circuit(construction_pass.run(circuit_to_dag(tcost_qc)))

# Now transpile to basis gates
generic_pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
init  = generic_pm.init
init.remove(3)
generic_pm.init = init
# generic_pm.layout = None
t_qaoa_circ = generic_pm.run(qaoa_circ)

print_circuit_info(t_qaoa_circ, 'QAOA circuit')
logger.info(t_qaoa_circ.count_ops())
logger.info(f'QAOA circuit has {t_qaoa_circ.num_qubits} qubits')


qaoa_depth = len(t_qaoa_circ.parameters) // 2
if init_type == 'ramp':
    t = 0.7 * p
    betas = np.linspace(
        (1 / p) * (t * (1 - 0.5 / p)), (1 / p) * (t * 0.5 / p), p
    )
    gammas = betas[::-1]
    init_params = betas.tolist() + gammas.tolist()
elif init_type == 'warm':
    raise Exception('Warm start not implemented')
else:
    init_params = rng.uniform(0, 1, qaoa_depth).tolist() + rng.uniform(0, 1, qaoa_depth).tolist()
logger.info(f'Init: {init_params}')


logger.info(f'Noise model: {getattr(backend.options, "noise_model", "Ideal noise")}')

history = []

def cvar(energies, alpha=1.0):
    """Compute the Conditional Value-at-Risk (CVaR) of a list of energies.

    Returns the mean of the lowest ``alpha`` fraction of energies, focusing
    the objective signal on the lowest-energy tail of the distribution.

    Args:
        energies: Sequence of energy values (floats).
        alpha: Tail fraction in (0, 1].  alpha=1.0 returns the plain mean;
            alpha=0.25 returns the mean of the bottom 25%.

    Returns:
        CVaR estimate as a float.
    """
    sorted_energies = sorted(energies)
    end_idx = int(alpha * len(energies))
    return np.sum(sorted_energies[0:end_idx]) / end_idx


def objective(x: np.ndarray):
    """Evaluate the CVaR objective for a given parameter vector.

    Submits the parameterised QAOA circuit to the IBM Sampler, collects
    bitstring counts, evaluates each bitstring against the remapped full
    Hamiltonian, and returns the CVaR of the resulting energy distribution.
    Appends a record to the module-level ``history`` list.

    Args:
        x: 1-D parameter array of length 2*p, ordered as
           [beta_0, ..., beta_{p-1}, gamma_0, ..., gamma_{p-1}].

    Returns:
        CVaR energy (float) to be minimised.
    """
    start = time()
    assigned_circuit = t_qaoa_circ.assign_parameters(x, inplace=False)
    sampler_job = sampler.run([assigned_circuit], shots=shots)
    sampler_result = sampler_job.result()
    counts = sampler_result[0].data.c.get_counts()
    sampling_time = time() - start
    start = time()
    energies = []
    evals = evaluate_sparse_pauli_samples(counts.keys(), remapped_full_hamiltonian)
    energies = [count * [evals[idx]] for idx, count in enumerate(counts.values())]
    flat_energies = [x for xs in energies for x in xs]
    total_energy = cvar(flat_energies, args.alpha)

    classical_post_process_time = time() - start
    history.append((sampling_time, total_energy, x.tolist(), counts, classical_post_process_time))
    return total_energy


def callback(intermediate_result: OptimizeResult):
    """Log the current optimiser state; raise StopIteration if optimal found.

    Used with gradient-based scipy optimisers that pass an OptimizeResult.

    Args:
        intermediate_result: Current optimiser state provided by scipy.

    Raises:
        StopIteration: When the objective reaches -1 (used as an early-exit
            sentinel).
    """
    logger.info(f'Current params: {intermediate_result.x}. Current func value: {intermediate_result.fun}')
    if intermediate_result.fun == -1:
        raise StopIteration


def callback_cobyla(xk: np.ndarray):
    """Log the current parameter vector for COBYLA/SLSQP/TNC optimisers.

    Args:
        xk: Current parameter vector provided by the optimiser.
    """
    logger.info(f'Current params: {xk}.')
    
    
logger.info(f'Using method: {args.method}.')

with Session(backend=backend):
    sampler = Sampler()
    result = minimize(
        objective, 
        x0=init_params, 
        method=args.method, 
        bounds=tuple((0,1) for _ in range(2 * p)), 
        options={"maxiter": 100, "maxfev": 100},
        callback=callback if args.method not in ['SLSQP', 'COBYLA', 'TNC'] else callback_cobyla
    )
logger.info(result)

obj_to_dump = dict(
    result=result, history=history, remapped_full_hamiltonian=remapped_full_hamiltonian, t_qaoa_circ=t_qaoa_circ, compiled_hamiltonian=compiled_hamiltonian, layout=layout
)

dump_file = basepath + filename.replace('compilation', 'optimisation') + '.method{}.cvar{}.p{}.shots{}.init{}.d{}'.format(
    args.method, args.alpha, p,shots, init_type, swap_depth
) + '.pkl'
with open(dump_file, 'wb') as f:
    pickle.dump(obj_to_dump, f)
