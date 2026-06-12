"""CVaR-QAOA parameter optimisation on an MPS-backed AerSimulator.

Implements Conditional Value-at-Risk (CVaR) QAOA: instead of minimising the
mean energy expectation ⟨H⟩, the objective is the mean energy of the lowest-α
fraction of measurement outcomes (CVaR_α).  Focusing on low-energy tails
improves convergence towards high-quality combinatorial solutions.

The circuit is constructed via SAT-mapped SWAP routing (qopt-best-practices),
which reduces 2-qubit gate depth by exploiting the commutativity structure of
the cost Hamiltonian.

CLI usage::

    python optimize.py -f <filename> [-p <reps>] [-N <nodes>] [-m <memory>]
                       [-M <method>] [-n <shots>] [-a <alpha>]
                       [--hardware] [--noisy] [--init {ramp,random,fixed}]

Args:
    -f / --filename (str): Base name of the QUBO data ``.pkl`` file.
    -p / --reps (int): QAOA circuit depth (default: 4).
    -N / --nodes (int): Number of graph nodes (optional, for Grover mixer init).
    -m / --memory (int): Simulator memory limit in MB (default: 4000).
    -M / --method (str): Aer simulation method (default: ``''``).
    -n / --shots (int): Shots per objective evaluation (default: 2000).
    -a / --alpha (float): CVaR threshold — fraction of lowest-energy samples
        used in the objective (default: 0.25).
    --hardware: Use the FakeFez backend noise model instead of ideal simulation.
    --noisy: Attach noise model to the Sampler.
    --init: Parameter initialisation strategy — ``"ramp"``, ``"random"``,
        or ``"fixed"`` (default: ``"random"``).

Output:
    Saves a pickle file containing the optimisation result, full history,
    circuit graph artefacts, best parameters, and best samples to the
    experiments output directory.
"""

import numpy as np
from time import time
import pickle
from scipy.optimize import minimize, OptimizeResult
import argparse
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeFez

from qopt_best_practices.sat_mapping import SATMapper

from qiskit_qaoa.utils.circuit_graph_utils import circuit_to_graph, graph_to_operator, circuit_construction
from qiskit_qaoa.utils.hamiltonian_utils import get_Q_and_hamiltonian
from qiskit_qaoa.utils.string_utils import evaluate_sparse_pauli_samples
from qiskit_qaoa.utils.logging import get_logger

logger = get_logger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename')
parser.add_argument('-p', '--reps', type=int, default=4)
parser.add_argument('-N', '--nodes', nargs='?', type=int)
parser.add_argument('-m', '--memory', type=int, default=4000)
parser.add_argument('-M', '--method', type=str, default='')
parser.add_argument('-n', '--shots', type=int, default=2000)
parser.add_argument('-a', '--alpha', type=float, default=0.25)
parser.add_argument('--hardware', action='store_true', default=False)
parser.add_argument('--noisy', action='store_true', default=False)
parser.add_argument('--init', choices=['ramp', 'random', 'fixed'], default='random')
args = parser.parse_args()


logger.info(args)

filename = args.filename
p: int = args.reps
hardware = args.hardware
shots = args.shots
noisy = args.noisy
init_type = args.init
alpha = args.alpha

seed = 1
rng = np.random.default_rng()

backend_options = dict(
    method='matrix_product_state',
    matrix_product_state_max_bond_dimension='20', 
    device='GPU',
    precision='single'
)
fake_fez = FakeFez()
backend = AerSimulator.from_backend(fake_fez, **backend_options)

data_file = f'/lustre/scratch127/qpg/jc59/out/oriented/qubo_data_{filename}.gfa.pkl'

Q, hamiltonian, offset, ising_offset = get_Q_and_hamiltonian(data_file)
qc = QAOAAnsatz(
    cost_operator=hamiltonian,
    reps = p,
    flatten=True
)
transpiled_qc = transpile(qc, backend, optimization_level=3, seed_transpiler=seed)


def print_circuit_info(qc, circuit_name):
    """Log qubit count, 2-qubit gate count, and 2-qubit gate depth of a circuit.

    Args:
        qc: A Qiskit ``QuantumCircuit`` to inspect.
        circuit_name (str): Label to include in the log message.
    """
    logger.info(
        f'{circuit_name} has {qc.num_qubits} qubits, \
        {qc.count_ops().get("cz", 0) + qc.count_ops().get("rzz", 0) + qc.count_ops().get("cx", 0)} 2Q gates \
        and {qc.depth(lambda instr: len(instr.qubits) > 1)} 2Q depth'
    )


print_circuit_info(transpiled_qc, '(Transpiled) Circuit')

graph = circuit_to_graph(qc, qc.parameters[p])

swap_strat = SwapStrategy.from_line(range(graph.order()))
edge_coloring = {(idx, idx + 1): (idx + 1) % 2 for idx in range(graph.order())}

remapped_g, sat_map, min_sat_layers = SATMapper(timeout=30).remap_graph_with_sat(
    graph=graph, swap_strategy=swap_strat
)
if remapped_g is None:
    raise Exception('Failed to find initial layout')

cost_op = graph_to_operator(remapped_g)
singles = cost_op[cost_op.paulis.z.sum(axis=-1) == 1]
doubles = cost_op[cost_op.paulis.z.sum(axis=-1) == 2]

# init_state = QuantumCircuit(cost_op.num_qubits)
# theta = 2*np.arcsin((2*args.nodes+1)**-0.5 )
# init_state.rx(0.1, range(init_state.num_qubits))
init_state = None
circ_dict = circuit_construction(singles, doubles, backend, swap_strat, edge_coloring, {}, p, init_state=init_state)

backend_circ = circ_dict["backend"]
print_circuit_info(backend_circ, '(Transpiled) Remapped Circuit')


if hardware:
    # transpiled again for the FakeFez backend
    circuit: QuantumCircuit = circ_dict["backend"]
else:
    backend = AerSimulator(**backend_options)
    circuit: QuantumCircuit = circ_dict["circuit_to_sample"]

qaoa_depth = len(circuit.parameters) // 2


if init_type == 'ramp':
    t = 0.7 * p
    betas = np.linspace(
        (1 / p) * (t * (1 - 0.5 / p)), (1 / p) * (t * 0.5 / p), p
    )
    gammas = betas[::-1]
    init_params = betas.tolist() + gammas.tolist()
elif init_type == 'fixed':
    init_params = [2.69911474, 2.54850482]
else:
    init_params = rng.uniform(0, np.pi, qaoa_depth).tolist() + rng.uniform(-np.pi, np.pi, qaoa_depth).tolist()
logger.info(f'Init: {init_params}')

if noisy:
    sampler = Sampler.from_backend(backend=backend, seed=seed)
else:
    sampler = Sampler(seed=seed, options=dict(backend_options=backend_options))
logger.info(f'Noise model: {getattr(sampler._backend.options, "noise_model", "Ideal noise")}')

history = []
best_func_val = np.inf
best_params = init_params
best_samples = []

def callback(intermediate_result: OptimizeResult):
    """Log the current parameter vector and objective value (gradient-based solvers).

    Args:
        intermediate_result: scipy ``OptimizeResult`` with ``.x`` (current
            parameters) and ``.fun`` (current objective value).
    """
    logger.info(f'Current params: {intermediate_result.x}. Current func value: {intermediate_result.fun}')


def callback_cobyla(xk: np.ndarray):
    """Log the current parameter vector (COBYLA callback interface).

    Args:
        xk: Current parameter array passed by COBYLA at each iteration.
    """
    logger.info(f'Current params: {xk}.')


def cvar(energies, alpha=1.0):
    """Compute the Conditional Value-at-Risk (CVaR) of an energy distribution.

    Sorts the energy samples and returns the mean of the lowest-α fraction.
    CVaR_α == mean(E) when α == 1.0 (standard expectation value).

    Args:
        energies: Iterable of scalar energy values from circuit measurements.
        alpha (float): CVaR threshold in (0, 1].  Lower values concentrate the
            objective on the best-energy samples (default: 1.0).

    Returns:
        float: Mean energy of the ``floor(alpha * len(energies))`` lowest-
        energy samples.
    """
    sorted_energies = sorted(energies)
    end_idx = int(alpha * len(energies))
    return np.sum(sorted_energies[0:end_idx]) / end_idx


def objective(x: np.ndarray):
    """Evaluate the CVaR-QAOA objective for a given parameter vector.

    Runs the QAOA circuit with parameters ``x``, evaluates each sampled
    bitstring against the cost operator, and returns the CVaR of the resulting
    energy distribution.  Updates module-level best tracking and appends
    timing/energy data to ``history``.

    Args:
        x: 1-D parameter array (betas then gammas) of length ``2 * qaoa_depth``.

    Returns:
        float: CVaR_alpha of the sampled energy distribution.
    """
    start = time()
    assigned_circuit = circuit.assign_parameters(x, inplace=False)
    sampler_job = sampler.run([assigned_circuit], shots=shots)
    sampler_result = sampler_job.result()
    counts = sampler_result[0].data.c.get_counts()
    sampling_time = time() - start
    start = time()
    energies = []
    evals = evaluate_sparse_pauli_samples(counts.keys(), cost_op) + ising_offset
    # int_samples = [np.array([int(x) for x in sample[::-1]]) for sample in counts.keys()]
    # evals = np.array([
    #     sample @ Q @ sample for sample in int_samples
    # ]) + offset
    energies = [count * [evals[idx]] for idx, count in enumerate(counts.values())]
    flat_energies = [x for xs in energies for x in xs]
    total_energy = cvar(flat_energies, alpha)

    global best_func_val
    global best_params
    global best_samples
    if total_energy < best_func_val:
        best_func_val = total_energy
        best_params = x
        best_samples = counts

    classical_post_process_time = time() - start
    history.append((sampling_time, total_energy, x.tolist(), counts, classical_post_process_time))
    return total_energy

method = "COBYLA"
result = minimize(
    objective, x0=init_params, 
    method=method, 
    bounds=tuple((0, np.pi) for _ in range(p)) +tuple((-np.pi, np.pi) for _ in range(p)), 
    options={"maxiter": 120, "maxfev": 120, "rhobeg": 0.1, "ftol": 1e-12},  # 
    callback=callback if method not in ['SLSQP', 'COBYLA', 'TNC'] else callback_cobyla
)
logger.info(result)


obj_to_dump = dict(
    result=result, history=history, singles=singles, doubles=doubles, sat_map=sat_map, graph=graph, 
    cost_op=cost_op, best_func_val=best_func_val, best_params=best_params, best_samples=best_samples,
    circuit=circuit
)
with open(f'/lustre/scratch127/qpg/jc59/out/qiskit/experiments/{filename}_cvar.alpha{alpha}.p{p}.shots{shots}.method{method}.hardware{hardware}.noisy{noisy}.init{init_type}.pkl', 'wb') as f:
    pickle.dump(obj_to_dump, f)
