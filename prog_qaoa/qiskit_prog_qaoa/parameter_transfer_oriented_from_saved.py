import numpy as np
import pickle
from scipy.optimize import minimize
from collections import Counter
from fnmatch import fnmatch

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as Sampler

# from qiskit_algorithms.optimizers import SPSA # Breaks because QNSPSA wants base sampler V1, deprecated


from qiskit_prog_qaoa.utils.opt_utils import oriented_objective, callback, callback_cobyla, oriented_soln_to_path, oriented_cost_function
from qiskit_prog_qaoa.utils.oriented_graph_to_circuit import gfa_file_to_oriented_prog_qaoa_circuit
from qiskit_prog_qaoa.utils.argparser import get_parser
from qiskit_prog_qaoa.utils.logging import get_logger

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def print_circuit_info(qc: QuantumCircuit, circuit_name):
    logger.info(f'{circuit_name} has {qc.num_qubits} qubits')
    logger.info(f'{circuit_name} has {qc.num_nonlocal_gates()} non-local gates and {qc.depth(lambda instr: len(instr.qubits) > 1)} non-local depth')
    logger.info(f'{circuit_name} contains {list(qc.count_ops().keys())} gates.')


logger = get_logger(__name__)
parser = get_parser()
parser.add_argument('-t', '--transfer')
args = parser.parse_args()

transfer = args.transfer
filename = args.filename
p: int = args.reps
shots = args.shots
init_type = args.init
max_iter = args.maxiter
method: str = args.method
lamda = args.lamda
blocking = args.blocking

seed = 1
rng = np.random.default_rng(seed=seed)

backend_options = dict(
    method='statevector',
    device='GPU',
    max_memory_mb=args.memory*0.9,
    cuStateVec_enable=True,
    blocking_qubits=blocking,
    precision='single'
)

backend = AerSimulator(**backend_options)
sampler = Sampler(options=dict(backend_options=backend_options))


with open(f'/tmp/jc59/out/prog_qaoa/oriented/{transfer}.p{p}.shots1000.init{init_type}.method{method}.iter{max_iter}.pkl', 'rb') as f:
    data = pickle.load(f)

init_params = data['result'].x
logger.info(f'Init: {init_params}')


large_circuit, large_n, large_K, large_T, large_graph = gfa_file_to_oriented_prog_qaoa_circuit(f'/tmp/jc59/data/{filename}.gfa', p, lamda)

d_circuit = large_circuit.decompose(gates_to_decompose=['state_prep', 'phase_operator', 'mixer_operator'], reps=1)
gtd = ['circuit*']
while any(fnmatch(key, p) for p in gtd for key in d_circuit.count_ops().keys()):
    d_circuit = d_circuit.decompose(gates_to_decompose=gtd)

print_circuit_info(d_circuit, 'Large Circuit')

t_circuit = transpile(d_circuit, backend, optimization_level=3, seed_transpiler=seed)
print_circuit_info(t_circuit, 'Transpiled Large Circuit')

t_circuit.measure_all()

large_history=[]
logger.info(f'Opt method: {method}')

if method == 'none':
    exit(0)
elif method == 'spsa':
    raise Exception('SPSA algorithm from qiskit_algorithms does not support Qiskit 2.0')
    # spsa = SPSA(maxiter=max_iter, termination_checker=TerminationChecker())
    # result = spsa.minimize(objective, x0=init_params)
    # print(f'SPSA completed after {result.nit} iterations')
else:
    large_result = minimize(
        oriented_objective, 
        x0=init_params,
        args=(large_n, large_T, large_graph, lamda, shots, large_history, t_circuit, sampler), 
        method=method, 
        bounds=tuple((0,1) for _ in range(2 * p)), 
        options={"maxiter": 10, "maxfev": 10, "rhobeg": 0.01},  # , "ftol": 1e-7
        callback=callback if method not in ['SLSQP', 'COBYLA', 'TNC'] else callback_cobyla
    )

logger.info(large_result)
to_run = t_circuit.assign_parameters(init_params)
sample = backend.run(to_run, shots=shots).result()


obj_to_dump = dict(
    result=large_result, sample=sample,
    history=large_history, init_params=init_params, lamda=lamda,
    circuit=large_circuit, graph=large_graph, n=large_n, T=large_T, K=large_K
)
# with open(f'/lustre/scratch127/qpg/jc59/out/prog_qaoa/oriented/{filename}.p{p}.shots{shots}.init{init_type}.method{method}.iter{max_iter}.pkl', 'wb') as f:
with open(f'/tmp/jc59/out/prog_qaoa/oriented/transfer.{transfer}.{filename}.p{p}.shots{shots}.init{init_type}.method{method}.iter{max_iter}.pkl', 'wb') as f:
    pickle.dump(obj_to_dump, f)


counts = Counter(sample.get_counts())
most_common = counts.most_common(100)
for e in most_common:
    if e[1] > 1:
        logger.info(f'soln: {e[0]}. path: {oriented_soln_to_path(e[0], large_n, large_T, large_graph)}. \
        cost: {oriented_cost_function(e[0], large_n, large_T, large_graph, lamda)}. count: {e[1]}')

exit(0)