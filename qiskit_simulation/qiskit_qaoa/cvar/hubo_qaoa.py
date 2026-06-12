"""CVaR-QAOA applied to a HUBO (Higher-Order Unconstrained Binary Optimisation) cost Hamiltonian.

Unlike standard QUBO, where the cost function contains at most pairwise
interactions, HUBO allows higher-order terms (degree > 2) in the polynomial
objective.  This module constructs the HUBO cost Hamiltonian directly from a
pangenome GFA file by:

1. Building a directed graph of segment nodes and copy-number weights.
2. Formulating a binary polynomial objective (path-cover constraint + copy-
   number matching) using symbolic arithmetic (sympy).
3. Converting the polynomial to an Ising (Z-basis) Hamiltonian via the
   substitution x_i = 0.5 - Z_i/2.
4. Applying SAT-mapped SWAP routing and the QAOA ansatz swap-strategy pass
   manager to compile an efficient p=1 circuit.
5. Reporting 2-qubit gate counts and depths for multiple transpilation levels.

CLI usage::

    python hubo_qaoa.py -f <filename> -c <copy_numbers>

Args:
    -f / --filename (str): Base name (without path/extension) of the GFA file
        under ``/nfs/.../pangenome/data/<filename>.gfa``.
    -c / --copy-numbers (str): Comma-separated list of float copy numbers,
        one per GFA segment.
"""

import numpy as np
import re
import gfapy
import networkx as nx
import argparse


from qiskit import transpile, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz, PauliEvolutionGate

from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy
from qopt_best_practices.sat_mapping import SATMapper

from qopt_best_practices.transpilation import qaoa_swap_strategy_pm


from sympy import Poly, Symbol

from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeFez


from qiskit_qaoa.utils.logging import get_logger

class Binary(Symbol):
    """A sympy Symbol that squares to itself, modelling binary (0/1) variables.

    Overrides ``_eval_power`` so that ``x**k == x`` for any power ``k``,
    which is the idempotent property of binary variables (x^2 = x).
    """

    def _eval_power(self, other):
        return self


def monomial_to_pauli(monomial):
    """Convert a sympy monomial of Binary symbols to a Z-basis Pauli string.

    Identifies all variable indices appearing in the monomial (from symbol
    names of the form ``Z[i]``) and places a ``'Z'`` at the corresponding
    positions in an all-``'I'`` Pauli string of length ``n * T``.

    Args:
        monomial: A sympy expression whose atoms are ``Binary`` symbols with
            names parseable as integers via regex.

    Returns:
        str: A Pauli string of length ``n * T`` with ``'Z'`` at each qubit
        index present in the monomial and ``'I'`` elsewhere.
    """
    indices = [int(re.search(r'[0-9]+', atom.name).group(0)) for atom in monomial.atoms()]
    pauli_str = ['I'] * n * T
    for i in indices:
        pauli_str[i] = 'Z'
    return ''.join(pauli_str)


def two_qubit_count(qc):
    """Count the total number of 2-qubit gates in a circuit.

    Counts CZ, RZZ, and CX gates.

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


def bin_rep(k):
    """Return the little-endian binary representation of integer ``k``.

    Args:
        k (int): Non-negative integer to represent.

    Returns:
        list[int]: List of ``n`` bits, least-significant bit first.
    """
    return [int(x) for x in np.binary_repr(k, n)[::-1]]


logger = get_logger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename')
parser.add_argument('-c', '--copy-numbers', help='delimited list input', 
    type=lambda s: [float(item) for item in s.split(',') if len(item)])
args = parser.parse_args()

logger.info(args)

filename = args.filename
copy_numbers = args.copy_numbers

seed = 1
rng = np.random.default_rng(seed=seed)

backend_options = dict(
    method='statevector',
    device='GPU',
    cuStateVec_enable=True,
    # blocking_enable=True,
    # blocking_qubits=24,
    precision='single'
)
fake_fez = FakeFez()

backend = AerSimulator.from_backend(fake_fez, **backend_options)

filepath = f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/data/{filename}.gfa'

gfa = gfapy.Gfa.from_file(filepath, vlevel=0)

# if filename[0:4] == 'test':
#     ratio = 1
# else:
#     ratio = 100

graph = nx.DiGraph()
for index, segment_line in enumerate(gfa.segments):
    graph.add_node(f'{segment_line.name}_+', weight=copy_numbers[index], start=segment_line.st)
    graph.add_node(f'{segment_line.name}_-', weight=copy_numbers[index], start=segment_line.st)
for edge_line in gfa.edges:
    v1 = edge_line.sid1
    v2 = edge_line.sid2
    graph.add_edges_from([
        (f'{v1.name}_{v1.orient}', f'{v2.name}_{v2.orient}'),
    ])
    v1.invert()
    v2.invert()
    graph.add_edges_from([
        (f'{v2.name}_{v2.orient}', f'{v1.name}_{v1.orient}'),
    ])

nodes = list(graph.nodes)
N = len(nodes)
n = int(np.ceil(np.log2(N+1)))
total_weight = int(sum(graph.nodes[node]["weight"] for node in nodes) / 2)
T = int(1.1 * total_weight)

x = [[Binary(f'x[{t}][{i}]') for i in range(n)] for t in range(T)]


logger.info(N)
logger.info(T)

constraint = sum([
    1 - sum([
        np.prod([
            1 - x[t][k] - bin_rep(i)[k] + 2 * x[t][k] * bin_rep(i)[k]
        for k in range(n)]) * sum([
            np.prod([
                1 - x[t+1][k] - bin_rep(j)[k] + 2 * x[t+1][k] * bin_rep(j)[k]
            for k in range(n)])
        for j in [nodes.index(nbr) for nbr in graph.neighbors(nodes[i])]])
    for i in range(N)])
for t in range(T-1)])

logger.info("Computed constraint terms")


obj = sum([
    (
        sum([
            np.prod([
                1 - x[t][k] - bin_rep(i)[k] + 2 * x[t][k] * bin_rep(i)[k]
            for k in range(n)])
            + np.prod([
                1 - x[t][k] - bin_rep(i+1)[k] + 2 * x[t][k] * bin_rep(i+1)[k]
            for k in range(n)])
        for t in range(T)])
        - graph.nodes[nodes[i]]["weight"]
    ) ** 2
for i in range(0,N,2)])
    
logger.info("Computed objective terms")

lamda = 10
total = lamda * constraint + obj
# total = Poly(total)

logger.info("Computed total")


Z = [Binary(f"Z[{i}]") for i in range(n*T)]
ising = total.subs(zip([item for row in x for item in row], [0.5 - z/2 for z in Z]))

logger.info("Computed Ising model")


ising = Poly(ising, Z)

logger.info("Expanded Ising model")

# ising = ising.simplify()
ising_expr_coeffs = ising.as_expr().as_coefficients_dict()

logger.info("Computed ising coeffs")


hamiltonian = SparsePauliOp('I'*n*T, ising_expr_coeffs[1])
for (monomial, coeff) in ising_expr_coeffs.items():
    if monomial == 1:
        continue
    hamiltonian += SparsePauliOp(monomial_to_pauli(monomial), coeff)
hamiltonian = hamiltonian.sort(weight=True)

logger.info("Computed Hamiltonian")


# Should use a grover mixer ideally
qc = QAOAAnsatz(
    cost_operator=hamiltonian,
    reps = 1,
    flatten=True
)

logger.info("Computed QC")




# QAOA anasatz maps multi RZ to cnot constructions, breaking the 1 gate per qubit set rule
# 

# circuit_graph = circuit_to_graph(qc, qc.parameters[p])

edges = []
weights = []

order_2_edges = []
order_2_weights = []

for t in hamiltonian:   
    if np.sum(t.paulis[0].z) == 2:
        order_2_edges.append(np.nonzero(t.paulis[0].z)[0])
        order_2_weights.append(t.coeffs[0])
        
            
order_2_circuit_graph = nx.Graph()
for idx in range(len(order_2_edges)):
    order_2_circuit_graph.add_edge(order_2_edges[idx][0], order_2_edges[idx][1], weight=order_2_weights[idx])

logger.info(order_2_circuit_graph.nodes)
logger.info(order_2_circuit_graph.edges)

swap_strat = SwapStrategy.from_line(list(range(order_2_circuit_graph.order())))
edge_coloring = {(idx, idx + 1): (idx + 1) % 2 for idx in range(order_2_circuit_graph.order())}


remapped_g, sat_map, min_sat_layers = SATMapper(timeout=60).remap_graph_with_sat(
    graph=order_2_circuit_graph, swap_strategy=swap_strat
)

logger.info(sat_map)
logger.info(min_sat_layers)


# Use sat_map to re-order the qubits in the Hamiltonian
remapped_hamiltonian = hamiltonian.apply_layout([sat_map[i] for i in range(hamiltonian.num_qubits)])

# cost_op = graph_to_operator(remapped_g)
remapped_qc = QAOAAnsatz(
    cost_operator=remapped_hamiltonian,
    reps = 1,
    flatten=True
)

# Run QAOA strategy pm?
properties = {}

def get_permutation(pass_, dag, time, property_set, count):
    """Transpiler callback that captures the virtual-to-physical qubit mapping.

    Registered as a pass-manager callback to record the
    ``virtual_permutation_layout`` after the layout pass completes.  The
    result is stored in the module-level ``properties`` dict.

    Args:
        pass_: The transpiler pass that just ran.
        dag: The current DAG circuit.
        time: Elapsed time for the pass.
        property_set: Dictionary of properties set by transpiler passes.
        count: Pass execution count.
    """
    properties["virtual_permutation_layout"] = property_set["virtual_permutation_layout"]
    
    
config = {
    "num_layers": 1,
    "swap_strategy": swap_strat,
    "edge_coloring": edge_coloring,
    "construct_qaoa": False,
}
pm = qaoa_swap_strategy_pm(config)

doubles = remapped_hamiltonian[remapped_hamiltonian.paulis.z.sum(axis=-1) == 2]
rest = remapped_hamiltonian[remapped_hamiltonian.paulis.z.sum(axis=-1) != 2]
logger.info(f'Doubles: {len(doubles)}')
logger.info(f'Rest: {len(rest)}')


num_qubits = len(doubles[0].paulis[0])
doubles_circ = QAOAAnsatz(
    doubles,
    initial_state=QuantumCircuit(num_qubits),
    mixer_operator=QuantumCircuit(num_qubits)
)
tdoubles_circ = pm.run(doubles_circ, callback=get_permutation)


rest_circ = QuantumCircuit(num_qubits)
rest_circ.append(PauliEvolutionGate(rest, time=tdoubles_circ.parameters[0]), range(num_qubits))

trest = transpile(rest_circ)
cost_circ = trest.compose(tdoubles_circ, inplace=False)

t_cost_circ = transpile(cost_circ, optimization_level=3)

trqc = transpile(remapped_qc, optimization_level=3)

tqc = transpile(qc, optimization_level=3)


logger.info(f'Two qubit count QC: {two_qubit_count(qc)}')
logger.info(f'Two qubit depth QC: {depth(qc)}')
logger.info(f'QC ops: {qc.count_ops()}')



logger.info(f'Two qubit count TQC: {two_qubit_count(tqc)}')
logger.info(f'Two qubit depth TQC: {depth(tqc)}')
logger.info(f'TQC ops: {tqc.count_ops()}')


logger.info(f'Two qubit count RQC: {two_qubit_count(remapped_qc)}')
logger.info(f'Two qubit depth RQC: {depth(remapped_qc)}')
logger.info(f'RQC ops: {remapped_qc.count_ops()}')


logger.info(f'Two qubit count TRQC: {two_qubit_count(trqc)}')
logger.info(f'Two qubit depth TRQC: {depth(trqc)}')
logger.info(f'TRQC ops: {trqc.count_ops()}')

logger.info(f'Two qubit count cost_circ: {two_qubit_count(cost_circ)}')
logger.info(f'Two qubit depth cost_circ: {depth(cost_circ)}')
logger.info(f'cost_circ ops: {cost_circ.count_ops()}')

logger.info(f'Two qubit count t_cost_circ: {two_qubit_count(t_cost_circ)}')
logger.info(f'Two qubit depth t_cost_circ: {depth(t_cost_circ)}')
logger.info(f't_cost_circ ops: {t_cost_circ.count_ops()}')