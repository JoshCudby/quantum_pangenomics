
import numpy as np
import pickle
import argparse
from collections import Counter

from qiskit import transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_ibm_runtime.options import SamplerOptions, TwirlingOptions, DynamicalDecouplingOptions


from qopt_best_practices.sat_mapping import SATMapper

from qiskit_qaoa.utils.circuit_graph_utils import circuit_to_graph, graph_to_operator, circuit_construction
from qiskit_qaoa.utils.hamiltonian_utils import get_Q_and_hamiltonian
from qiskit_qaoa.utils.logging import get_logger

logger = get_logger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename')
parser.add_argument('-p', '--reps', type=int, default=4)
parser.add_argument('-n', '--shots', type=int, default=2000)
parser.add_argument('--init', choices=['ramp', 'random', 'fixed', 'warm'], default='random')
args = parser.parse_args()

logger.info(args)

filename = args.filename
p: int = args.reps
shots = args.shots
init_type = args.init

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
backend = service.backend(name='ibm_aachen')
logger.info(f'Backend: {backend}')
logger.info(f'Num qubits in backend: {backend.configuration().to_dict()["n_qubits"]}')


transpiled_qc = transpile(qc, backend, optimization_level=3)


def print_circuit_info(qc, circuit_name):
    logger.info(f'{circuit_name} has {qc.count_ops().get("cz", 0) + qc.count_ops().get("rzz", 0) + qc.count_ops().get("cx", 0) + qc.count_ops().get("swap", 0)+ qc.count_ops().get("ecr", 0)} 2Q gates \
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

circ_dict = circuit_construction(singles, doubles, backend, swap_strat, edge_coloring, {}, p)

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
    init_params = [0.439759390571553, 0.47523912591426465, 0.40683448432677694, 0.8665276969749175, 
                   0.07727284052271921, 0.25521259972136145, 0.04596182583685282, 0.04562939468439431]
    logger.info('Using fixed init values')
elif init_type == 'warm':
    if p == 1:
        # test_N4_W6 sweep
        init_params = [2.35810456, 2.81522484]   
    elif p == 2:
        init_params = [ 1.03193062,  0.66187895,  0.0063484 , -0.01153315]
    elif p == 3:
        # test_N2_W2
        init_params = [ 6.63183340e-01,  9.75245289e-01,  9.53929681e-01,  2.13708775e-02, 2.51865429e-02, -1.03691343e-04]
    elif p == 4:
        init_params = [0.97347025, 0.67623981, 0.79089901, 0.36223065, 0.13285854,
       0.64307387, 0.22996854, 0.44763577]
    else:
        raise Exception(f'Warm values not available for p = {p}')
    logger.info('Using warm init values')
else:
    init_params = rng.uniform(0, 1, qaoa_depth).tolist() + rng.uniform(0, 1, qaoa_depth).tolist()
logger.info(f'Init: {init_params}')


def cvar(energies, alpha=1.0):
    sorted_energies = sorted(energies)
    end_idx = int(alpha * len(energies))
    return np.sum(sorted_energies[0:end_idx]) / end_idx

sampler = Sampler(mode=backend)

error_miti = True
ddOptions = DynamicalDecouplingOptions(enable=False, sequence_type="XX")
twirlingOptions = TwirlingOptions(enable_gates=error_miti, enable_measure=error_miti, num_randomizations='auto', shots_per_randomization='auto', strategy="active-accum")
samplerOptions = SamplerOptions(dynamical_decoupling=ddOptions, twirling=twirlingOptions)
sampler = Sampler(mode=backend, options=samplerOptions)


assigned_circuit = circuit.assign_parameters(init_params, inplace=False)

sampler_job = sampler.run([assigned_circuit], shots=shots)
sampler_result = sampler_job.result()
counts = sampler_result[0].data.c.get_counts()
int_samples = [np.array([int(x) for x in sample[::-1]]) for sample in counts.keys()]
evals = np.array([
    sample @ Q @ sample for sample in int_samples
]) + offset
energies = [count * [evals[idx]] for idx, count in enumerate(counts.values())]
flat_energies = [x for xs in energies for x in xs]
total_energy = cvar(flat_energies, 1)
counter = Counter(flat_energies)

logger.info(f'Energy: {total_energy}')
logger.info(f'Best sample: {min(flat_energies)}')
logger.info(counter)

obj_to_dump = dict(
    singles=singles, doubles=doubles, sat_map=sat_map, graph=graph, 
    cost_op=cost_op, counts=counts, energy=total_energy, init_params=init_params
)
with open(f'/lustre/scratch127/qpg/jc59/out/qiskit/experiments/hardware.{filename}_sample.error_miti{error_miti}.p{p}.shots{shots}.init{init_type}.pkl', 'wb') as f:
    pickle.dump(obj_to_dump, f)
