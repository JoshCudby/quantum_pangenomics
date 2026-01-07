
import numpy as np
from time import time
import pickle
import argparse
from scipy.optimize import minimize, OptimizeResult

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy

from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2 as Sampler
from qiskit_ibm_runtime.options import SamplerOptions, TwirlingOptions, DynamicalDecouplingOptions

from qopt_best_practices.sat_mapping import SATMapper

from qiskit_qaoa.utils.string_utils import evaluate_sparse_pauli_samples
from qiskit_qaoa.utils.circuit_graph_utils import circuit_to_graph, graph_to_operator, circuit_construction
from qiskit_qaoa.utils.hamiltonian_utils import get_Q_and_hamiltonian
from qiskit_qaoa.utils.logging import get_logger

logger = get_logger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename')
parser.add_argument('-p', '--reps', type=int, default=4)
parser.add_argument('-a', '--alpha', type=float, default=0.25)
parser.add_argument('-N', '--nodes', type=int)
parser.add_argument('-M', '--method', type=str, default='COBYLA')
parser.add_argument('-n', '--shots', type=int, default=2000)
parser.add_argument('--init', choices=['ramp', 'random', 'fixed'], default='random')
args = parser.parse_args()

logger.info(args)

filename = args.filename
p: int = args.reps
shots = args.shots
init_type = args.init
alpha = args.alpha
N: int = args.nodes

rng = np.random.default_rng()

data_file = f'/lustre/scratch127/qpg/jc59/out/oriented/qubo_data_{filename}.gfa.pkl'

Q, hamiltonian, offset, ising_offset = get_Q_and_hamiltonian(data_file)
qc = QAOAAnsatz(
    cost_operator=hamiltonian,
    reps = p,
    flatten=True
)
num_qubits = hamiltonian.num_qubits

service = QiskitRuntimeService(name='eu_test_instance')
# backend = service.least_busy(min_num_qubits=num_qubits, operational=True, simulator=False) 
backend = service.backend(name='ibm_aachen')
logger.info(f'Backend: {backend}')
logger.info(f'Num qubits in backend: {backend.configuration().to_dict()["n_qubits"]}')


transpiled_qc = transpile(qc, backend, optimization_level=3)


def print_circuit_info(qc, circuit_name):
    logger.info(f'{circuit_name} has {qc.count_ops().get("cz", 0) + qc.count_ops().get("rzz", 0) + qc.count_ops().get("cx", 0) + qc.count_ops().get("ecr", 0)} 2Q gates \
    and {qc.depth(lambda instr: len(instr.qubits) > 1)} 2Q depth')


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

init_state = QuantumCircuit(cost_op.num_qubits)
theta = 2*np.arcsin((2*(N+1))**-0.5 )
init_state.rx(0.1, range(init_state.num_qubits))
circ_dict = circuit_construction(singles, doubles, backend, swap_strat, edge_coloring, {}, p, init_state=init_state)

circuit = circ_dict["backend"]
print_circuit_info(circuit, '(Transpiled) Remapped Circuit')


qaoa_depth = len(circuit.parameters) // 2


if init_type == 'ramp':
    t = 0.7 * p
    betas = np.linspace(
        (1 / p) * (t * (1 - 0.5 / p)), (1 / p) * (t * 0.5 / p), p
    )
    gammas = betas[::-1]
    init_params = betas.tolist() + gammas.tolist()
elif init_type == 'fixed':
    raise Exception('No fixed values provided')
    logger.info('Using fixed init values')
else:
    init_params = rng.uniform(0, np.pi, qaoa_depth).tolist() + rng.uniform(-np.pi, np.pi, qaoa_depth).tolist()
logger.info(f'Init: {init_params}')

history = []
best_func_val = np.inf
best_params = init_params
best_samples = []

def callback(intermediate_result: OptimizeResult):
    logger.info(f'Current params: {intermediate_result.x}. Current func value: {intermediate_result.fun}')
    

def callback_cobyla(xk: np.ndarray):
    logger.info(f'Current params: {xk}.')


def cvar(energies, alpha=1.0):
    sorted_energies = sorted(energies)
    end_idx = int(alpha * len(energies))
    return np.sum(sorted_energies[0:end_idx]) / end_idx


def objective(x: np.ndarray):
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


method = args.method
max_iter = 120
with Session(backend=backend):
    ddOptions = DynamicalDecouplingOptions(enable=False, sequence_type="XX")
    twirlingOptions = TwirlingOptions(enable_gates=True, enable_measure=True, num_randomizations='auto', shots_per_randomization='auto', strategy="active-accum")
    samplerOptions = SamplerOptions(dynamical_decoupling=ddOptions, twirling=twirlingOptions)
    sampler = Sampler(options=samplerOptions)
    logger.info(sampler.options)
    result = minimize(
        objective, 
        x0=init_params, 
        method=method, 
        bounds=tuple((0, np.pi) for _ in range(p)) +tuple((-np.pi, np.pi) for _ in range(p)),
        options={"maxiter": max_iter, "maxfev": 120, "rhobeg": 0.1},
        callback=callback if args.method not in ['SLSQP', 'COBYLA', 'TNC'] else callback_cobyla
    )
logger.info(result)


obj_to_dump = dict(
    result=result, history=history, singles=singles, doubles=doubles, sat_map=sat_map, graph=graph, 
    cost_op=cost_op, best_func_val=best_func_val, best_params=best_params, best_samples=best_samples
)
with open(f'/lustre/scratch127/qpg/jc59/out/qiskit/cvar_new/hardware/{filename}_cvar.error_miti.alpha{alpha}.p{p}.shots{shots}.method{method}.max_iter{max_iter}.init{init_type}.pkl', 'wb') as f:
    pickle.dump(obj_to_dump, f)
