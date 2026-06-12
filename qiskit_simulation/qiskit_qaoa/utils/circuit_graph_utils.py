"""Circuit-to-graph conversions and QAOA ansatz construction.

Provides utilities for extracting the weighted interaction graph from a QAOA
cost-layer circuit, converting that graph back to a ``SparsePauliOp``
operator, and building the full multi-layer QAOA circuit (cost + mixer + initial
state) with optional backend-specific transpilation.
"""

import copy
import networkx as nx
from qiskit.quantum_info import SparsePauliOp
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QAOAAnsatz, PauliEvolutionGate
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qopt_best_practices.transpilation.qaoa_construction_pass import QAOAConstructionPass
from qopt_best_practices.transpilation import qaoa_swap_strategy_pm

from qiskit_qaoa.utils.logging import get_logger

logger = get_logger(__name__)


def circuit_to_graph(qc: QuantumCircuit, parameter) -> nx.Graph:
    """Extract the weighted interaction graph from a QAOA cost-layer circuit.

    Iterates over the circuit's instruction data and collects every gate whose
    parameter expression depends on ``parameter``.  One- and two-qubit gates
    become self-loop and edge entries respectively.  The edge weight is derived
    by evaluating the parameter expression at ``parameter=1`` and dividing by 2.

    Args:
        qc: A QAOA cost-layer ``QuantumCircuit`` containing parameterised
            gates.
        parameter: The ``ParameterExpression`` (typically a ``Parameter``
            object named ``gamma``) to filter on.

    Returns:
        A ``networkx.Graph`` whose nodes are qubit indices and whose edges
        carry a ``weight`` attribute equal to the scaled gate coefficient.

    Raises:
        ValueError: If a gate in the circuit acts on more than two qubits.
        ValueError: If the same edge appears more than once in the circuit.
    """
    qreg = qc.qregs[0]
    graph, edges = nx.Graph(), []
    graph.add_nodes_from(range(len(qreg)))
    seen_edges = set()

    for inst in qc.data:
        iop = inst.operation

        if len(iop.params) == 0 or parameter not in iop.params[0].parameters:
            continue

        if len(inst.qubits) == 1:
            edge = (qreg.index(inst.qubits[0]), qreg.index(inst.qubits[0]))
        elif len(inst.qubits) == 2:
            edge = (qreg.index(inst.qubits[0]), qreg.index(inst.qubits[1]))
        else:
            raise ValueError('Too many qubits in instruction')
        
        if edge in seen_edges:
            logger.info(inst)
            raise ValueError(f'Circuit contains edge {edge} multiple times')

        # logger.info(edge)
        seen_edges.add(edge)
        seen_edges.add(edge[::-1])

        param_expression = copy.deepcopy(iop.params[0])
        param_expression = param_expression.assign(next(iter(param_expression.parameters)), 1)
        weight = float(param_expression) / 2.0
        edges.append((edge[0], edge[1], weight))

    graph.add_weighted_edges_from(edges)
    return graph


def graph_to_operator(graph: nx.Graph, physical_qubits=None, prefactor: float = 1.0) -> SparsePauliOp:
    """Convert a weighted interaction graph to a SparsePauliOp ZZ Hamiltonian.

    Each edge ``(i, j, weight)`` in the graph becomes a ``Z_i Z_j`` term with
    coefficient ``prefactor * weight``.  The Pauli strings are in little-endian
    Qiskit ordering (qubit 0 at rightmost character).

    Args:
        graph: A ``networkx.Graph`` with optional ``weight`` edge data.
            Nodes must be integer qubit indices.
        physical_qubits: Total number of qubits for the operator.  Defaults
            to ``len(graph)`` (i.e. the number of nodes).
        prefactor: A global scalar multiplied into every coefficient (default 1.0).

    Returns:
        A ``SparsePauliOp`` representing the ZZ Ising Hamiltonian.
    """
    pauli_list = []
    if physical_qubits is None:
        physical_qubits = len(graph)
    for node1, node2, data in graph.edges(data=True):
        paulis = ["I"] * physical_qubits
        paulis[node1], paulis[node2] = "Z", "Z"
        if "weight" in data:
            weight = data["weight"] 
        else:
            logger.info("No weight data")
            weight = 1.0
        pauli_list.append(("".join(paulis)[::-1], prefactor * weight))

    return SparsePauliOp.from_list(pauli_list)


def circuit_construction(
        singles,
        doubles,
        backend,
        swap_strat,
        edge_coloring,
        metadata,
        reps,
        init_state=None,
        mixer_layer=None
):
    """Build the full multi-layer QAOA circuit from single-qubit and two-qubit Hamiltonians.

    The doubles (ZZ) Hamiltonian is compiled using the swap-strategy pass
    manager; the singles (single-qubit Z) Hamiltonian is transpiled to RZ gates
    and prepended to form the combined cost layer.  The ``QAOAConstructionPass``
    then appends the initial state and mixer layers and handles layout-aware
    measurement assignment.  Optionally, the resulting circuit is also
    transpiled for a specific backend at optimisation level 3.

    Args:
        singles: A ``SparsePauliOp`` containing only single-qubit Z terms of
            the Hamiltonian.
        doubles: A ``SparsePauliOp`` containing only ZZ (two-qubit) terms.
        backend: A Qiskit backend to transpile for, or ``None`` to skip
            backend-specific transpilation.
        swap_strat: An ``ExtendedSwapStrategy`` (or compatible) swap strategy
            for routing ZZ gates.
        edge_coloring: Edge-coloring dict for the swap strategy coupling map.
        metadata: Metadata dict attached to the backend-compiled circuit.
        reps: Number of QAOA layers (p).
        init_state: Optional ``QuantumCircuit`` for the initial state.
            Defaults to an empty circuit (all-zero state).
        mixer_layer: Optional ``QuantumCircuit`` for the mixer layer.
            Defaults to the standard X-rotation mixer.

    Returns:
        A dict with the following keys:

        - ``'doubles'``: The QAOAAnsatz circuit built from ``doubles`` only.
        - ``'tdoubles'``: The swap-strategy-compiled doubles circuit.
        - ``'cost_circuit'``: Singles prepended to ``tdoubles``.
        - ``'circuit_to_sample'``: Full QAOA circuit ready for sampling.
        - ``'backend'`` (only if ``backend`` is not ``None``): The backend-
          transpiled circuit with ``metadata`` attached.
    """
    circuits_dict = {}
    n = len(doubles[0].paulis[0])

    doubles_circ = QAOAAnsatz(
        doubles,
        initial_state=QuantumCircuit(n),
        mixer_operator=QuantumCircuit(n)
    )
    circuits_dict['doubles'] = doubles_circ

    properties = {}

    def get_permutation(pass_, dag, time, property_set, count):
        properties["virtual_permutation_layout"] = property_set["virtual_permutation_layout"]

    
    config = {
        "num_layers": reps,
        "swap_strategy": swap_strat,
        "edge_coloring": edge_coloring,
        "construct_qaoa": False,
        # "basis_gates": ["sx", "x", "rz", "rzz", "swap", "cx", "id"]
    }
    pm = qaoa_swap_strategy_pm(config)
    tdoubles_circ = pm.run(doubles_circ, callback=get_permutation)
    circuits_dict["tdoubles"] = tdoubles_circ

    singles_circ = QuantumCircuit(n)
    singles_circ.append(PauliEvolutionGate(singles, time=tdoubles_circ.parameters[0]), range(n))
    tsingles = transpile(singles_circ, basis_gates=["rz"])
    cost_circ = tsingles.compose(tdoubles_circ, inplace=False)
    circuits_dict["cost_circuit"] = cost_circ

    construction_pass = QAOAConstructionPass(reps, init_state=init_state, mixer_layer=mixer_layer)
    construction_pass.property_set = properties
    transpiled_circ = dag_to_circuit(construction_pass.run(circuit_to_dag(cost_circ)))

    circuits_dict["circuit_to_sample"] = transpiled_circ

    if backend is not None:
        generic_pm = generate_preset_pass_manager(optimization_level=3, backend=backend, scheduling_method="alap")
        circuits_dict["backend"] = generic_pm.run(transpiled_circ)
        circuits_dict["backend"].metadata = metadata

    return circuits_dict


