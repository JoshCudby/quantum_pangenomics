
import numpy as np
import argparse
import pickle
from scipy.optimize import minimize, OptimizeResult, basinhopping

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
parser.add_argument('--init', choices=['ramp', 'random', 'fixed', 'warm'], default='random')
args = parser.parse_args()

logger.info(args)

filename = args.filename
p: int = args.reps
init_type = args.init

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


if init_type == 'ramp':
    t = 0.7 * p
    betas = np.linspace(
        (1 / p) * (t * (1 - 0.5 / p)), (1 / p) * (t * 0.5 / p), p
    )
    gammas = betas[::-1]
    init_params = betas.tolist() + gammas.tolist()
elif init_type == 'warm':
    if p == 4:
        init_params = [ 9.33323444e-01,  7.08009649e-03,  7.36344025e-01,  9.37923754e-01,
            2.29973290e-02,  2.75044958e-03,  8.34971625e-04, -2.92071056e-04]
    elif p == 2:
        init_params = [0.86782694, 0.99261561, 0.02069131, 0.84516142]
    else:
        raise Exception('No warm values available')
    print('Using warm init values')
else:
    init_params = rng.uniform(0, 1, qaoa_depth).tolist() + rng.uniform(0, 1, qaoa_depth).tolist()
print(f'Init: {init_params}')


history = []
best_func_val = np.inf
best_params = init_params
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

    history.append((energy, x.tolist(), sv))
    return energy


# method = "COBYLA"
# result = minimize(
#     objective, x0=init_params, 
#     method=method, 
#     bounds=tuple((0,1) for _ in range(2 * p)), 
#     options={"maxiter": 10000, "maxfev": 10000, "rhobeg": 0.01,},  #  "ftol": 1e-7
#     callback=callback if method not in ['SLSQP', 'COBYLA', 'TNC'] else callback_cobyla
# )
# logger.info(result)

method = "basinhopping"
min_method="COBYLA"
result = basinhopping(
    objective, 
    x0=np.array(init_params), 
    niter=1000,
    callback=callback_basinhopping,
    minimizer_kwargs=dict(
        method=min_method, 
        bounds=tuple((0,1) for _ in range(2 * p)), 
        options={"maxiter": 5000, "maxfev": 1000, "rhobeg": 0.1,},  #  "ftol": 1e-7
    )
)
logger.info(result)


obj_to_dump = dict(
    result=result, history=history, singles=singles, doubles=doubles, sat_map=sat_map, graph=graph, 
    cost_op=cost_op, best_func_val=best_func_val, best_params=best_params, best_sv=best_sv
)
with open(f'/lustre/scratch127/qpg/jc59/out/qiskit/cvar_new/{filename}_cvar.p{p}.method{method}.no_shot_noise.init{init_type}.pkl', 'wb') as f:
    pickle.dump(obj_to_dump, f)
