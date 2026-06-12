"""Updated CVaR-QAOA optimisation variant (statevector GPU backend).

An updated version of ``optimize.py`` that switches the Aer backend to a GPU
statevector simulator (with cuStateVec support) and loads the Hamiltonian
directly from a pre-built pickle file rather than deriving it from a GFA QUBO
data file.  The COBYLA objective is fixed at α = 0.05.

CLI usage::

    python optimize2.py -p <reps> [-m <memory>] [-M <method>] [-n <shots>]
                        [--hardware] [--noisy] [--init {ramp,random}]

Args:
    -p / --reps (int): QAOA circuit depth (default: 4).
    -m / --memory (int): Simulator memory limit in MB (default: 4000).
    -M / --method (str): Aer simulation method (unused in statevector path).
    -n / --shots (int): Shots per objective evaluation (default: 2000).
    --hardware: Route through the FakeFez backend transpilation.
    --noisy: Attach noise model to the Sampler.
    --init: Parameter initialisation strategy — ``"ramp"`` or ``"random"``
        (default: ``"random"``).

Output:
    Saves a pickle containing the optimisation result and history to
    ``/lustre/.../orson/phylo.cvar.<suffix>.pkl``.
"""

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
from qiskit_qaoa.utils.string_utils import evaluate_sparse_pauli_samples
from qiskit_qaoa.utils.argparser import get_parser
from qiskit_qaoa.utils.logging import get_logger

logger = get_logger(__name__)
parser = get_parser()
args = parser.parse_args()

logger.info(args)

p: int = args.reps
hardware = args.hardware
shots = args.shots
noisy = args.noisy
init_type = args.init

seed = 1
rng = np.random.default_rng(seed=seed)

backend_options = dict(
    method='statevector',
    device='GPU',
    max_memory_mb=args.memory*0.9,
    cuStateVec_enable=True,
    # blocking_enable=True,
    # blocking_qubits=24,
    # batched_shots_gpu_max_qubits=24,
    # batched_shots_gpu=noisy,
    precision='single'
)
fake_fez = FakeFez()
backend = AerSimulator.from_backend(fake_fez, **backend_options)

data_file = '/nfs/users/nfs_j/jc59/quantumwork/pangenome/data/26_qubit_ham.pickle'
with open(data_file, 'rb') as f:
    hamiltonian = pickle.load(f)
    
qc = QAOAAnsatz(
    cost_operator=hamiltonian,
    reps = p,
    flatten=True
)
transpiled_qc = transpile(qc, backend, optimization_level=3, seed_transpiler=seed)


def print_circuit_info(qc, circuit_name):
    """Log the 2-qubit gate count and 2-qubit gate depth of a circuit.

    Args:
        qc: A Qiskit ``QuantumCircuit`` to inspect.
        circuit_name (str): Label to include in the log message.
    """
    logger.info(f'{circuit_name} has {qc.count_ops().get("cz", 0) + qc.count_ops().get("rzz", 0) + qc.count_ops().get("cx", 0)} 2Q gates \
    and {qc.depth(lambda instr: len(instr.qubits) > 1)} 2Q depth')


print_circuit_info(transpiled_qc, '(Transpiled) Circuit')

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
else:
    init_params = rng.uniform(0, 0.9 * np.pi, qaoa_depth).tolist() + rng.uniform(0, 0.5 * np.pi, qaoa_depth).tolist()
logger.info(f'Init: {init_params}')

if noisy:
    sampler = Sampler.from_backend(backend=backend, seed=seed)
else:
    sampler = Sampler(seed=seed, options=dict(backend_options=backend_options))
logger.info(f'Noise model: {getattr(sampler._backend.options, "noise_model", "Ideal noise")}')

history = []


def cvar(energies, alpha=1.0):
    """Compute the Conditional Value-at-Risk (CVaR) of an energy distribution.

    Returns the mean of the lowest-α fraction of energy samples.  Clamps the
    index to at least 1 to avoid division by zero on very small sample sets.

    Args:
        energies: Iterable of scalar energy values from circuit measurements.
        alpha (float): CVaR threshold in (0, 1].  Default is 1.0 (full mean).

    Returns:
        float: Mean energy of the ``max(1, floor(alpha * len(energies)))``
        lowest-energy samples.
    """
    sorted_energies = sorted(energies)
    end_idx = max(int(alpha * len(energies)), 1)
    return np.sum(sorted_energies[0:end_idx]) / end_idx


def objective(x: np.ndarray):
    """Evaluate the CVaR-QAOA objective for a given parameter vector.

    Runs the QAOA circuit with parameters ``x``, evaluates each bitstring
    against the cost operator (without the QUBO offset), and returns the CVaR
    at α = 0.05.  Timing and sample data are appended to ``history``.

    Args:
        x: 1-D parameter array (betas then gammas) of length
            ``2 * qaoa_depth``.

    Returns:
        float: CVaR_0.05 of the sampled energy distribution.
    """
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
    objective, x0=init_params, method="COBYLA", options={"maxiter": 200, "rhobeg": 0.1}
)
logger.info(result)


obj_to_dump = dict(
    result=result, history=history, singles=singles, doubles=doubles, sat_map=sat_map, graph=graph
)
with open(f'/lustre/scratch127/qpg/jc59/out/orson/phylo.cvar.p{p}.shots{shots}.hardware{hardware}.init{init_type}.pkl', 'wb') as f:
    pickle.dump(obj_to_dump, f)
