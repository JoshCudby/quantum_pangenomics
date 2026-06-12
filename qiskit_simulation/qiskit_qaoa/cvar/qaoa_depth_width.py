"""Circuit depth and width analysis for CVaR QAOA circuits.

For a given QUBO data file and circuit depth ``p``, this script:

1. Constructs the Ising Hamiltonian from the QUBO matrix.
2. Transpiles the raw QAOAAnsatz circuit to a generic heavy-hex backend
   (17 qubits, heavy-hex coupling map).
3. Applies SAT-mapped SWAP routing from qopt-best-practices.
4. Reports and appends to a results file: qubit count, ``T_max``, graph
   vertex count ``V``, and the 2-qubit gate counts and depths for both the
   naively transpiled and SWAP-mapped circuits.

Intended to be called in a batch loop over many problem instances to build
up a ``results.txt`` table for scaling analysis.

CLI usage::

    python qaoa_depth_width.py -f <filename> [-p <reps>] [-m <memory>]
                               [-d <data-dir>]

Args:
    -f / --filename (str): Base name of the QUBO data ``.pkl`` file.
    -p / --reps (int): QAOA circuit depth (default: 4).
    -m / --memory (int): Simulator memory limit in MB (default: 4000).
    -d / --data-dir (str): Directory containing the QUBO ``.pkl`` files
        (default: ``/lustre/.../oriented``).

Output:
    Appends a single CSV line to ``/lustre/.../qaoa_depth_width/results.txt``
    and logs the same line.
"""

import numpy as np
import pickle
import argparse
from typing import Tuple

from qiskit.providers.fake_provider.generic_backend_v2 import GenericBackendV2
from qiskit import transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy
from qiskit_optimization import QuadraticProgram, QiskitOptimizationError
from qiskit.quantum_info import SparsePauliOp, Pauli

from qiskit_aer import AerSimulator

from qopt_best_practices.sat_mapping import SATMapper

from qiskit_qaoa.utils.circuit_graph_utils import circuit_to_graph, graph_to_operator, circuit_construction
from qiskit_qaoa.utils.logging import get_logger


def mod_to_ising(quad_prog: QuadraticProgram) -> Tuple[SparsePauliOp, float]:
    """Return the Ising Hamiltonian of this problem.
    Args:
        quad_prog: The problem to be translated.

    Returns:
        A tuple (qubit_op, offset) comprising the qubit operator for the problem
        and offset for the constant value in the Ising Hamiltonian.

    Raises:
        QiskitOptimizationError: If an integer variable or a continuous variable exists
            in the problem.
        QiskitOptimizationError: If constraints exist in the problem.
    """
    # if problem has variables that are not binary, raise an error
    if quad_prog.get_num_vars() > quad_prog.get_num_binary_vars():
        raise QiskitOptimizationError(
            "The type of all variables must be binary. "
            "You can use `QuadraticProgramToQubo` converter "
            "to convert integer variables to binary variables. "
            "If the problem contains continuous variables, `to_ising` cannot handle it. "
            "You might be able to solve it with `ADMMOptimizer`."
        )

    # if constraints exist, raise an error
    if quad_prog.linear_constraints or quad_prog.quadratic_constraints:
        raise QiskitOptimizationError(
            "There must be no constraint in the problem. "
            "You can use `QuadraticProgramToQubo` converter "
            "to convert constraints to penalty terms of the objective function."
        )

    # initialize Hamiltonian.
    num_vars = quad_prog.get_num_vars()
    qubit_op = SparsePauliOp('I'* num_vars, 0)
    offset = 0.0
    zero = np.zeros(num_vars, dtype=bool)

    # set a sign corresponding to a maximized or minimized problem.
    # sign == 1 is for minimized problem. sign == -1 is for maximized problem.
    sense = quad_prog.objective.sense.value

    # convert a constant part of the objective function into Hamiltonian.
    offset += quad_prog.objective.constant * sense

    # create Pauli terms
    coeff_dict = {}
    for (i, j), coeff in quad_prog.objective.quadratic.to_dict().items():
        weight = coeff * sense / 4
        if i == j:
            offset += 2 * weight
            coeff_dict[(i,)] = coeff_dict.get((i,), 0) - 2 * weight
        else:
            offset += weight
            coeff_dict[(i,)] = coeff_dict.get((i,), 0) - weight
            coeff_dict[(j,)] = coeff_dict.get((j,), 0) - weight
            coeff_dict[(i, j)] = coeff_dict.get((i, j), 0) + weight

    def key_to_pauli(key: tuple):
        """Convert a tuple of variable indices to a Pauli Z operator.

        Args:
            key: Tuple of integer variable indices where Z operators should
                be placed.

        Returns:
            Pauli: A Pauli operator with Z at each index in ``key`` and I
            elsewhere.
        """
        z_p = np.zeros(num_vars, dtype=bool)
        for idx in key:
            z_p[idx] = True
        return Pauli((z_p, zero))
    qubit_op = SparsePauliOp(
        [key_to_pauli(key) for key in coeff_dict.keys()],
        [coeff for coeff in coeff_dict.values()]
    )
    return qubit_op, offset

logger = get_logger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename')
parser.add_argument('-p', '--reps', type=int, default=4)
parser.add_argument('-m', '--memory', type=int, default=4000)
parser.add_argument('-d', '--data-dir', default="/lustre/scratch127/qpg/jc59/out/oriented")
args = parser.parse_args()

logger.info(args)

filename = args.filename
p: int = args.reps

seed = 1
rng = np.random.default_rng(seed=seed)

backend_options = dict(
    method='statevector',
    # device='GPU',
    max_memory_mb=args.memory*0.9,
    precision='single'
)
coupling_map = CouplingMap.from_heavy_hex(17)
generic_backend = GenericBackendV2(num_qubits=len(coupling_map.physical_qubits), coupling_map=coupling_map)
backend = AerSimulator.from_backend(generic_backend, **backend_options)

backend.set_option('coupling_map', coupling_map)

data_file = f'{args.data_dir}/qubo_data_{filename}.pkl'

with open(data_file, 'rb') as f:
    data = pickle.load(f)
logger.info('Loaded data')
Q = np.array(data['Q'])
offset = data['offset']
Q = np.triu(Q) * 2
Q -= np.triu(np.triu(Q).T) / 2

normalisation = np.max(np.abs(Q))
Q = Q / normalisation
offset = offset / normalisation

logger.info(Q.shape)
logger.info(offset)

mod = QuadraticProgram("QUBO test")
mod.binary_var_list(Q.shape[0])
mod.minimize(constant=offset, linear=None, quadratic=Q)
hamiltonian, ising_offset = mod_to_ising(mod)
logger.info('Mapped to ising hamiltonian')
hamiltonian = hamiltonian.sort(weight=True)

qc = QAOAAnsatz(
    cost_operator=hamiltonian,
    reps = p,
    flatten=True
)
logger.info('Built qc')
transpiled_qc = transpile(qc, backend, optimization_level=3, seed_transpiler=seed)
logger.info('Transpiled qc')
graph = circuit_to_graph(qc, qc.parameters[p]) # Why 4??
logger.info('Built circuit DAG')

swap_strat = SwapStrategy.from_line(range(graph.order()))
edge_coloring = {(idx, idx + 1): (idx + 1) % 2 for idx in range(graph.order())}

remapped_g, sat_map, min_sat_layers = SATMapper(timeout=5).remap_graph_with_sat(
    graph=graph, swap_strategy=swap_strat
)
logger.info('SAT Mapped')

cost_op = graph_to_operator(remapped_g)
singles = cost_op[cost_op.paulis.z.sum(axis=-1) == 1]
doubles = cost_op[cost_op.paulis.z.sum(axis=-1) == 2]

circ_dict = circuit_construction(singles, doubles, backend, swap_strat, edge_coloring, {}, p)
logger.info('Built circuits')

backend_circ = circ_dict["backend"]


def two_qubit_count(qc):
    """Count the total number of CZ, RZZ, and CX gates in a circuit.

    Args:
        qc: A Qiskit ``QuantumCircuit``.

    Returns:
        int: Total 2-qubit gate count.
    """
    return qc.count_ops().get("cz", 0) + qc.count_ops().get("rzz", 0) + qc.count_ops().get("cx", 0)


def depth(qc):
    """Compute the 2-qubit gate depth of a circuit.

    Args:
        qc: A Qiskit ``QuantumCircuit``.

    Returns:
        int: Depth considering only gates acting on 2 or more qubits.
    """
    return qc.depth(lambda instr: len(instr.qubits) > 1)

logger.info(f'{Q.shape[0]}, {data["T_max"]}, {data["V"]}, {two_qubit_count(transpiled_qc)}, {depth(transpiled_qc)}, {two_qubit_count(backend_circ)}, {depth(backend_circ)}')
with open('/lustre/scratch127/qpg/jc59/qaoa_depth_width/results.txt', 'a') as f:
    f.write(f'{Q.shape[0]}, {data["T_max"]}, {data["V"]}, {two_qubit_count(transpiled_qc)}, {depth(transpiled_qc)}, {two_qubit_count(backend_circ)}, {depth(backend_circ)}\n')
