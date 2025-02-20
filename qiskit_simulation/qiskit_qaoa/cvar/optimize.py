
import numpy as np
from time import time
import pickle
from scipy.optimize import minimize

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy

from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as Sampler, EstimatorV2 as Estimator

from qiskit_ibm_runtime.fake_provider import FakeFez


from qopt_best_practices.sat_mapping import SATMapper


from qiskit_qaoa.utils.circuit_graph_utils import circuit_to_graph, graph_to_operator, circuit_construction
from qiskit_qaoa.utils.hamiltonian_utils import get_objective_and_hamiltonian
from qiskit_qaoa.utils.string_utils import evaluate_sparse_pauli_samples
from qiskit_qaoa.utils.logging import get_logger

logger = get_logger(__name__)
seed = 1
rng = np.random.default_rng(seed=seed)
p = 4

backend_options = dict(
    method='statevector',
    device='GPU',
    max_memory_mb=16000*0.9,
)
fake_fez = FakeFez()
backend = AerSimulator.from_backend(fake_fez, **backend_options)

filename = 'small_test'
data_file = f'/lustre/scratch127/qpg/jc59/out/tangle/qubo_data_{filename}.gfa.npy'

_, hamiltonian = get_objective_and_hamiltonian(data_file)
qc = QAOAAnsatz(
    cost_operator=hamiltonian,
    reps = p,
    flatten=True
)
transpiled_qc = transpile(qc, backend, optimization_level=3, seed_transpiler=seed)
logger.info(f'(Transpiled) Circuit has {transpiled_qc.count_ops().get("cz", 0) + transpiled_qc.count_ops().get("rzz", 0) + transpiled_qc.count_ops().get("cx", 0)} 2Q gates and {transpiled_qc.depth(lambda instr: len(instr.qubits) > 1)} 2Q depth')


graph = circuit_to_graph(qc, qc.parameters[p]) # Why 4??

swap_strat = SwapStrategy.from_line(range(graph.order()))
edge_coloring = {(idx, idx + 1): (idx + 1) % 2 for idx in range(graph.order())}

remapped_g, sat_map, min_sat_layers = SATMapper(timeout=60).remap_graph_with_sat(
    graph=graph, swap_strategy=swap_strat
)

cost_op = graph_to_operator(remapped_g)
singles = cost_op[cost_op.paulis.z.sum(axis=-1) == 1]
doubles = cost_op[cost_op.paulis.z.sum(axis=-1) == 2]

circ_dict = circuit_construction(singles, doubles, backend, swap_strat, edge_coloring, {}, p)


backend_circ = circ_dict["backend"]
logger.info(f'(Transpiled) Remapped Circuit has {backend_circ.count_ops().get("cz", 0) + backend_circ.count_ops().get("rzz", 0) + backend_circ.count_ops().get("cx", 0)} 2Q gates and {backend_circ.depth(lambda instr: len(instr.qubits) > 1)} 2Q depth')


# Sanity check
# test_vals = rng.uniform(0, 0.1 * np.pi, 2*p)
# test_tcirc = circ_dict["circuit_to_sample"].assign_parameters(test_vals, inplace=False)
# original_circ = qc.assign_parameters(test_vals, inplace=False)
# estimator = Estimator(options=dict(backend_options=backend_options))
# cost_op_pre_sat = graph_to_operator(graph)
# test_res = estimator.run([(test_tcirc, cost_op), (original_circ, cost_op_pre_sat)], precision=0.001).result()
# logger.info(test_res[0].data.evs)
# logger.info(test_res[1].data.evs)


hardware = True

if hardware:
    # transpiled again for the FakeFez backend
    circuit: QuantumCircuit = circ_dict["backend"]
else:
    backend = AerSimulator(**backend_options)
    circuit: QuantumCircuit = circ_dict["circuit_to_sample"]

qaoa_depth = len(circuit.parameters) // 2

init_params = rng.uniform(0, 0.9 * np.pi, qaoa_depth).tolist() + rng.uniform(0, 0.5 * np.pi, qaoa_depth).tolist()
shots = 2000


sampler = Sampler(seed=seed, options=dict(backend_options=backend_options))
history = []


def cvar(energies, alpha=1.0):
    sorted_energies = sorted(energies)
    end_idx = max(int(alpha * len(energies)), 1)
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
    evals = evaluate_sparse_pauli_samples(counts.keys(), cost_op)
    energies = [count * [evals[idx]] for idx, count in enumerate(counts.values())]
    flat_energies = [x for xs in energies for x in xs]
    total_energy = cvar(flat_energies, 0.05)

    classical_post_process_time = time() - start
    history.append((sampling_time, total_energy, x.tolist(), counts, classical_post_process_time))
    return total_energy

result = minimize(
    objective, x0=init_params, method="COBYLA", options={"maxiter": 100, "rhobeg": 0.1}
)
logger.info(result)


obj_to_dump = dict(
    result=result, history=history, singles=singles, doubles=doubles, sat_map=sat_map, graph=graph
)
with open(f'/lustre/scratch127/qpg/jc59/out/qiskit/cvar/{filename}_cvar.p{p}.hardware{hardware}.pkl', 'wb') as f:
    pickle.dump(obj_to_dump, f)
