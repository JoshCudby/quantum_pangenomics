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
    """"QAOA Cost operator as a circuit to a graph"""
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
            raise ValueError(f'Circuit contains edge {edge} multiple times')

        seen_edges.add(edge)
        seen_edges.add(edge[::-1])

        param_expression = copy.deepcopy(iop.params[0])
        param_expression = param_expression.assign(next(iter(param_expression.parameters)), 1)
        weight = float(param_expression) / 2.0
        edges.append((edge[0], edge[1], weight))

    graph.add_weighted_edges_from(edges)
    return graph


def graph_to_operator(graph: nx.Graph, prefactor: float = 1.0) -> SparsePauliOp:
    pauli_list = []
    for node1, node2, data in graph.edges(data=True):
        paulis = ["I"] * len(graph)
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
        reps
):
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
    }
    pm = qaoa_swap_strategy_pm(config)
    tdoubles_circ = pm.run(doubles_circ, callback=get_permutation)
    circuits_dict["tdoubles"] = tdoubles_circ

    singles_circ = QuantumCircuit(n)
    singles_circ.append(PauliEvolutionGate(singles, time=tdoubles_circ.parameters[0]), range(n))
    tsingles = transpile(singles_circ, basis_gates=["rz"])
    cost_circ = tsingles.compose(tdoubles_circ, inplace=False)
    circuits_dict["cost_circuit"] = cost_circ

    construction_pass = QAOAConstructionPass(reps)
    construction_pass.property_set = properties
    transpiled_circ = dag_to_circuit(construction_pass.run(circuit_to_dag(cost_circ)))

    circuits_dict["circuit_to_sample"] = transpiled_circ

    if backend is not None:
        generic_pm = generate_preset_pass_manager(optimization_level=3, backend=backend, scheduling_method="alap")
        circuits_dict["backend"] = generic_pm.run(transpiled_circ)
        circuits_dict["backend"].metadata = metadata

    return circuits_dict


