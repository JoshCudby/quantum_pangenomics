"""Shot-noise-free grid sweep over initial parameters for CVaR-QAOA.

Combines the parameter-grid sweep strategy from ``optimize_sweep.py`` with the
exact statevector energy computation from ``optimize_no_shot_noise.py``.  The
full statevector is saved after each circuit execution and the expected energy
is evaluated as a probability-weighted inner product with pre-computed basis
energies — eliminating shot noise entirely.

A Cartesian grid over (beta, gamma) is automatically reduced in resolution until
the total number of starting points is at most 100, then each grid point is
locally refined with the chosen scipy method.

CLI usage::

    python optimize_no_shot_noise_sweep.py -f <filename> [-p <reps>]
                                           [-M <method>]

Args:
    -f / --filename (str): Base name of the QUBO data ``.pkl`` file.
    -p / --reps (int): QAOA circuit depth (default: 4).
    -M / --method (str): scipy optimisation method (default: ``"COBYLA"``).

Output:
    Saves a pickle file containing the best result, full history, circuit
    graph artefacts, best statevector, and best parameters to the experiments
    output directory.
"""

import numpy as np
import argparse
import pickle
from itertools import product
from scipy.optimize import minimize, OptimizeResult

from qiskit import transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy
from qiskit_ibm_runtime.fake_provider import FakeFez

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
    """Print the 2-qubit gate count and 2-qubit gate depth of a circuit.

    Args:
        qc: A Qiskit ``QuantumCircuit`` to inspect.
        circuit_name (str): Label to include in the output.
    """
    print(f'{circuit_name} has {qc.count_ops().get("cz", 0) + qc.count_ops().get("rzz", 0) + qc.count_ops().get("cx", 0)} 2Q gates \
    and {qc.depth(lambda instr: len(instr.qubits) > 1)} 2Q depth')


data_file = f'/lustre/scratch127/qpg/jc59/out/oriented/qubo_data_{filename}.gfa.pkl'

Q, hamiltonian, offset, _ = get_Q_and_hamiltonian(data_file)


backend_options = dict(
    method='statevector',
    device='GPU',
    precision='single'
)
# backend = AerSimulator(**backend_options)
# backend.set_option("n_qubits", hamiltonian.num_qubits)
fake_fez = FakeFez()
backend = AerSimulator.from_backend(fake_fez, **backend_options)

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
best_res = None

def callback(intermediate_result: OptimizeResult):
    """Log the current parameter vector and objective value (gradient-based solvers).

    Args:
        intermediate_result: scipy ``OptimizeResult`` with ``.x`` and ``.fun``.
    """
    logger.info(f'Current params: {intermediate_result.x}. Current func value: {intermediate_result.fun}')


def callback_cobyla(xk: np.ndarray):
    """Log the current parameter vector (COBYLA callback interface).

    Args:
        xk: Current parameter array passed by COBYLA at each iteration.
    """
    logger.info(f'Current params: {xk}.')


def callback_basinhopping(x: np.ndarray, f: float, accept: bool):
    """Log the current parameters and objective value (basin-hopping callback).

    Args:
        x: Current parameter array.
        f: Current objective value.
        accept: Whether the step was accepted by the basin-hopping criterion.
    """
    logger.info(f'Current params: {x}. Current func value: {f}')


def cvar(energies, alpha=1.0):
    """Compute the Conditional Value-at-Risk (CVaR) of an energy distribution.

    Args:
        energies: Iterable of scalar energy values.
        alpha (float): CVaR threshold in (0, 1].  Default is 1.0 (full mean).

    Returns:
        float: Mean energy of the ``floor(alpha * len(energies))`` lowest-
        energy samples.
    """
    sorted_energies = sorted(energies)
    end_idx = int(alpha * len(energies))
    return np.sum(sorted_energies[0:end_idx]) / end_idx

int_samples = [np.array([int(x) for x in np.binary_repr(y, width=hamiltonian.num_qubits)]) for y in range(2**hamiltonian.num_qubits)]
evals = np.array([
    sample @ Q @ sample for sample in int_samples
]) + offset



def objective(x: np.ndarray):
    """Evaluate the shot-noise-free QAOA energy for a given parameter vector.

    Saves the full statevector after circuit execution and computes the
    expected energy as the probability-weighted sum of pre-computed basis-state
    energies (``evals``).  Updates module-level best tracking.

    Args:
        x: 1-D parameter array (betas then gammas) of length
            ``2 * qaoa_depth``.

    Returns:
        float: Exact expected energy ⟨ψ(x)|H|ψ(x)⟩ from the statevector.
    """
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


beta_resolution, gamma_resolution = 11, 11
while (beta_resolution ** p) * (gamma_resolution ** p) > 100:
    if beta_resolution == gamma_resolution:
        beta_resolution -= 1
    else:
        gamma_resolution -= 1
        
params = [
    x[0] + x[1] 
    for x in product(
        product(np.linspace(0 + np.pi/(2*beta_resolution), np.pi - np.pi/(2*beta_resolution), beta_resolution), repeat=p), 
        product(np.linspace(-np.pi + np.pi/(gamma_resolution), np.pi - np.pi/(gamma_resolution), gamma_resolution), repeat=p)
    )
]
logger.info(f'Number of runs: {len(params)}')
for idx, param in enumerate(params):
    res = minimize(
        objective, 
        x0=param, 
        method=args.method, 
        bounds=tuple((0, np.pi) for _ in range(p)) +tuple((-np.pi, np.pi) for _ in range(p)), 
        options={"maxiter": 100, "rhobeg": 1/(3*gamma_resolution)},  # , "ftol": 1e-7
        # callback=callback if args.method not in ['SLSQP', 'COBYLA', 'TNC'] else callback_cobyla
    )
    if res.fun == best_func_val:
        best_res = res
    if idx % 10 == 99:
        logger.info(f'Completed search number {idx}')


obj_to_dump = dict(
    best_result=best_res, history=history, singles=singles, doubles=doubles, sat_map=sat_map, graph=graph, 
    cost_op=cost_op, best_func_val=best_func_val, best_params=best_params, best_sv=best_sv, circuit=circuit
)
with open(f'/lustre/scratch127/qpg/jc59/out/qiskit/experiments/{filename}.sweep.p{p}.no_shot_noise.pkl', 'wb') as f:
    pickle.dump(obj_to_dump, f)
