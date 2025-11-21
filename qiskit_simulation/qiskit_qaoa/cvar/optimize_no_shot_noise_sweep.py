
import numpy as np
import argparse
import pickle
from itertools import product
from scipy.optimize import minimize, OptimizeResult

from qiskit import transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy

from qiskit_aer import AerSimulator


from qopt_best_practices.sat_mapping import SATMapper

from qiskit_qaoa.utils.circuit_graph_utils import circuit_to_graph, graph_to_operator, circuit_construction
from qiskit_qaoa.utils.hamiltonian_utils import get_Q_and_hamiltonian
from qiskit_qaoa.utils.logging import get_logger

logger = get_logger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename')
parser.add_argument('-p', '--reps', type=int, default=4)
parser.add_argument('-M', '--method', type=str, default='COBYLA')
args = parser.parse_args()

logger.info(args)

filename = args.filename
p: int = args.reps

seed = 1
rng = np.random.default_rng()

def print_circuit_info(qc, circuit_name):
    print(f'{circuit_name} has {qc.count_ops().get("cz", 0) + qc.count_ops().get("rzz", 0) + qc.count_ops().get("cx", 0)} 2Q gates \
    and {qc.depth(lambda instr: len(instr.qubits) > 1)} 2Q depth')
    

data_file = f'/lustre/scratch127/qpg/jc59/out/oriented/qubo_data_{filename}.gfa.pkl'

Q, hamiltonian, offset, _ = get_Q_and_hamiltonian(data_file)


backend_options = dict(
    method='statevector',
    device='GPU',
    precision='single'
)
backend = AerSimulator(**backend_options)
backend.set_option("n_qubits", hamiltonian.num_qubits)

qc = QAOAAnsatz(
    cost_operator=hamiltonian,
    reps = p,
    flatten=True
)
transpiled_qc = transpile(qc, backend, optimization_level=3, seed_transpiler=seed)
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

circ_dict = circuit_construction(singles, doubles, None, swap_strat, edge_coloring, {}, p)

circuit = circ_dict["circuit_to_sample"]
circuit.remove_final_measurements()
print_circuit_info(circuit, '(Transpiled) Remapped Circuit')

backend = AerSimulator(**backend_options)

qaoa_depth = len(circuit.parameters) // 2


history = []
best_func_val = np.inf
best_params = []
best_sv = np.array([])

def callback(intermediate_result: OptimizeResult):
    logger.info(f'Current params: {intermediate_result.x}. Current func value: {intermediate_result.fun}')
    

def callback_cobyla(xk: np.ndarray):
    logger.info(f'Current params: {xk}.')

def callback_basinhopping(x: np.ndarray, f: float, accept: bool):
    logger.info(f'Current params: {x}. Current func value: {f}')


def cvar(energies, alpha=1.0):
    sorted_energies = sorted(energies)
    end_idx = int(alpha * len(energies))
    return np.sum(sorted_energies[0:end_idx]) / end_idx

int_samples = [np.array([int(x) for x in np.binary_repr(y, width=hamiltonian.num_qubits)]) for y in range(2**hamiltonian.num_qubits)]
evals = np.array([
    sample @ Q @ sample for sample in int_samples
]) + offset


def objective(x: np.ndarray):
    assigned_circuit = circuit.assign_parameters(x, inplace=False)
    assigned_circuit.save_statevector()
    job = backend.run([assigned_circuit])
    result = job.result()
    data = result.results[0].data
    sv = np.asarray(data.statevector)
    energy = np.sum(np.abs(sv) ** 2 * evals)
    
    global best_func_val
    global best_params
    global best_sv
    if energy < best_func_val:
        best_func_val = energy
        best_params = x
        best_sv = sv

    history.append((energy, x.tolist()))
    return energy


resolution = 3
while resolution ** (2*p) < 1000:
    resolution += 1
results = []
for param in product(np.linspace(0.01, 0.99, resolution), repeat=2*p):
    results.append(minimize(
        objective, 
        x0=param, 
        method=args.method, 
        bounds=tuple((0,1) for _ in range(2 * p)), 
        options={"maxiter": 300, "rhobeg": 1/(2*resolution)},  # , "ftol": 1e-7
        # callback=callback if args.method not in ['SLSQP', 'COBYLA', 'TNC'] else callback_cobyla
    ))


obj_to_dump = dict(
    results=results, history=history, singles=singles, doubles=doubles, sat_map=sat_map, graph=graph, 
    cost_op=cost_op, best_func_val=best_func_val, best_params=best_params, best_sv=best_sv
)
with open(f'/lustre/scratch127/qpg/jc59/out/qiskit/cvar_new/{filename}.sweep.p{p}.no_shot_noise.pkl', 'wb') as f:
    pickle.dump(obj_to_dump, f)
