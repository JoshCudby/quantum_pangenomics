
import numpy as np
import pickle
import argparse
from scipy.optimize import minimize, OptimizeResult

from itertools import product

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate, CXGate
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import InverseCancellation, CommutativeCancellation


from qopt_best_practices.transpilation.swap_cancellation_pass import SwapToFinalMapping
from qopt_best_practices.transpilation.qaoa_construction_pass import QAOAConstructionPass

from qiskit_aer import AerSimulator
from qiskit_aer.backends.backendconfiguration import AerBackendConfiguration

from qiskit_qaoa.utils.commuting_gate_router import CommutingGateRouter
from qiskit_qaoa.utils.gfa_utils import gfa_file_to_graph
from qiskit_qaoa.hubo.graph_to_hubo_hamiltonian import graph_to_hubo_hamiltonian
from qiskit_qaoa.utils.transpiler_passes import ExtendedSwapStrategy, FindCommutingPauliEvolutionsMulti
from qiskit_qaoa.utils.string_utils import evaluate_sparse_pauli_samples
from qiskit_qaoa.utils.logging import get_logger



def print_circuit_info(qc, circuit_name):
    logger.info(f'{circuit_name} has {qc.count_ops().get("cz", 0) + qc.count_ops().get("rzz", 0) + qc.count_ops().get("cx", 0)} 2Q gates \
    and {qc.depth(lambda instr: len(instr.qubits) > 1)} 2Q depth')
    

logger = get_logger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename')
parser.add_argument('-M', '--method', type=str, default="COBYLA")
parser.add_argument('-p', '--reps', type=int, default=4)
parser.add_argument('-m', '--memory', type=int, default=16000)
parser.add_argument('-c', '--copy-numbers', help='delimited list input', 
    type=lambda s: [float(item) for item in s.split(',') if len(item)])

args = parser.parse_args()

logger.info(args)

filename: str = args.filename
p: int = args.reps
copy_numbers: list[float] = args.copy_numbers
seed = 1
rng = np.random.default_rng()

basis_gates=["sx", "x", "rz", "rzz", "cz", "id", "swap", "cx", "h"]

filepath = f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/data/{filename}.gfa'
graph, n, V, T = gfa_file_to_graph(filepath, copy_numbers)

full_hamiltonian = graph_to_hubo_hamiltonian(graph, n, T, lamda=10, constraint_terms=1.0)
num_qubits = n * T
extended_swap_strat = ExtendedSwapStrategy.from_all_to_all(n * T)

num_physical_qubits = extended_swap_strat._num_vertices
coupling_map = extended_swap_strat._coupling_map


backend_options = dict(
    method='statevector',
    device='GPU',
    memory=0.9*args.memory,
    precision='single',
    basis_gates=basis_gates,
)
config = AerSimulator._DEFAULT_CONFIGURATION
config["n_qubits"] = num_physical_qubits
config["basis_gates"] = basis_gates
config = AerBackendConfiguration.from_dict(config)
backend = AerSimulator(configuration=config, coupling_map=extended_swap_strat._coupling_map, **backend_options)
backend.set_option("n_qubits", num_physical_qubits)
logger.info(f'Num qubits in backend: {backend.configuration().to_dict()["n_qubits"]}')


pm_rz = PassManager(
    [
        FindCommutingPauliEvolutionsMulti(), 
        CommutingGateRouter(
            extended_swap_strat,
            max_layers=0,
            perform_extra_swaps=True
        ),
        SwapToFinalMapping(),
        InverseCancellation(gates_to_cancel=[CXGate()]),
        CommutativeCancellation(basis_gates=["cx", "swap", "rz", "rzz"]),
        InverseCancellation(gates_to_cancel=[CXGate()]),
    ]
)
qc = QuantumCircuit(num_physical_qubits)
qc.append(PauliEvolutionGate(full_hamiltonian, time=Parameter("c")), range(num_physical_qubits))   
tqc_rz = pm_rz.run(qc)
print_circuit_info(tqc_rz, 'Transpiled cost hamiltonian circuit')
print(tqc_rz.count_ops())
logger.info(f'Cost hamiltonian circuit has {tqc_rz.num_qubits} qubits')


construction_pass = QAOAConstructionPass(p)
t_qaoa_circ = dag_to_circuit(construction_pass.run(circuit_to_dag(tqc_rz)))
t_qaoa_circ.remove_final_measurements()

# Now transpile to basis gates
# generic_pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
# init  = generic_pm.init
# init.remove(3)
# generic_pm.init = init
# generic_pm.layout = None
# t_qaoa_circ = generic_pm.run(qaoa_circ)

# print_circuit_info(t_qaoa_circ, 'QAOA circuit')
# logger.info(t_qaoa_circ.count_ops())
# logger.info(f'QAOA circuit has {t_qaoa_circ.num_qubits} qubits')


qaoa_depth = len(t_qaoa_circ.parameters) // 2



history = []
best_func_val = np.inf
best_params = []
best_sv = np.array([])


keys = [np.binary_repr(x, num_physical_qubits) for x in range(2**num_physical_qubits)]
evals = evaluate_sparse_pauli_samples(keys, full_hamiltonian)
zero_eval_indexs = np.nonzero(evals == 0)


def callback_cobyla(xk: np.ndarray):
    logger.info(f'Current params: {xk}.')


def callback(intermediate_result: OptimizeResult):
    logger.info(f'Current params: {intermediate_result.x}. Current func value: {intermediate_result.fun}')



def objective(x: np.ndarray):
    assigned_circuit = t_qaoa_circ.assign_parameters(x, inplace=False)
    assigned_circuit.save_statevector()
    job = backend.run([assigned_circuit])
    result = job.result()
    data = result.results[0].data
    sv = np.asarray(data.statevector)
    energy = np.sum(np.abs(sv) ** 2 * evals)
    score = -np.sum(np.abs(sv[zero_eval_indexs]) ** 2)
    
    global best_func_val
    global best_params
    global best_sv
    if score < best_func_val:
        best_func_val = score
        best_params = x
        best_sv = sv

    history.append((energy, score, x.tolist()))
    # return energy
    return score


beta_resolution, gamma_resolution = 30, 30
while (beta_resolution ** p) * (gamma_resolution ** p) > 1000:
    if beta_resolution == gamma_resolution:
        beta_resolution -= 1
    else:
        gamma_resolution -= 1
        
results = []
params = [
    x[0] + x[1] 
    for x in product(
        product(np.linspace(0 + np.pi/(2*beta_resolution), np.pi - np.pi/(2*beta_resolution), beta_resolution), repeat=p), 
        product(np.linspace(-np.pi + np.pi/(gamma_resolution), np.pi - np.pi/(gamma_resolution), gamma_resolution), repeat=p)
    )
]

logger.info('Starting optimisation sweep')
for idx, param in enumerate(params):
    results.append(minimize(
        objective, 
        x0=param, 
        method=args.method, 
        bounds=tuple((0, np.pi) for _ in range(p)) +tuple((-np.pi, np.pi) for _ in range(p)), 
        options={"maxiter": 300, "rhobeg": 1/(3*gamma_resolution)},  # , "ftol": 1e-7
        # callback=callback if args.method not in ['SLSQP', 'COBYLA', 'TNC'] else callback_cobyla
    ))
    if idx % 100 == 99:
        logger.info(f'Completed search number {idx}')


obj_to_dump = dict(
    results=results, history=history, t_qaoa_circ=t_qaoa_circ, hamiltonian=full_hamiltonian, 
    best_sv=best_sv, best_func_val=best_func_val,best_params=best_params
)

dump_file = f'/lustre/scratch127/qpg/jc59/out/qiskit/hubo_no_shot_noise_optimum/sweep.{filename}.p{p}.pkl'
with open(dump_file, 'wb') as f:
    pickle.dump(obj_to_dump, f)
