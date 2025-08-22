from __future__ import annotations

import pickle
import networkx as nx
import numpy as np
import numpy.typing as npt
import math
from typing import Tuple
from collections.abc import Iterable
from itertools import combinations, permutations, product
from functools import reduce

from qiskit import QuantumCircuit

from qiskit.circuit import Gate, Qubit, Clbit
from qiskit.circuit.library import PauliEvolutionGate, CXGate, RZZGate
from qiskit.dagcircuit import DAGCircuit, DAGOpNode

from qiskit.transpiler import TransformationPass
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from collections import defaultdict

from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.converters import circuit_to_dag

from qiskit.exceptions import QiskitError

from qiskit_qaoa.utils.logging import get_logger

logger = get_logger(__name__)
rng = np.random.default_rng(seed=1)



class CommutingBlock(Gate):
    """A gate made of commuting two-qubit gates.

    This gate is intended for use with commuting swap strategies to make it convenient
    for the swap strategy router to identify which blocks of operations commute.
    """

    def __init__(self, node_block: Iterable[DAGOpNode]) -> None:
        """
        Args:
            node_block: A block of nodes that commute.

        Raises:
            QiskitError: If the nodes in the node block do not apply to two-qubits.
        """
        qubits: set[Qubit] = set()
        cbits: set[Clbit] = set()
        for node in node_block:
            # if len(node.qargs) != 2:
            #     raise QiskitError(f"Node {node.name} does not apply to two-qubits.")

            qubits.update(node.qargs)
            cbits.update(node.cargs)

        if cbits:
            raise QiskitError(
                f"{self.__class__.__name__} does not accept nodes with classical bits."
            )

        super().__init__(
            "commuting_block", num_qubits=len(qubits), params=[], label="Commuting gates"
        )
        self.node_block = node_block
        self.qubits = qubits

    def __iter__(self):
        """Iterate through the nodes in the block."""
        return iter(self.node_block)
    
    
class DecomposePauliZEvolution(TransformationPass):
    """Decomposes :class:`.PauliEvolutionGate`s, where the operators are all Z, on a coupling map."""
    def __init__(
        self,
        coupling_map: CouplingMap
    ) -> None:
        r"""
        Args:
            coupling_map: A CouplingMap from a backend or SwapStrategy
        """
        super().__init__()
        self._coupling_map = coupling_map
        
    
    def run(self, dag: DAGCircuit) -> DAGCircuit:
        new_dag = dag.copy_empty_like()
        for node in dag.topological_op_nodes():
            if isinstance(node.op, PauliEvolutionGate) and self.valid_operator(node, dag):
                sub_dag = self._decompose(dag, node)
                new_dag.compose(sub_dag)
            else:
                new_dag.apply_operation_back(node.op, node.qargs, node.cargs)
        return new_dag
        
        
    def valid_operator(self, node: DAGOpNode, dag: DAGCircuit) -> bool:
        """Determine if only a single Pauli consisting of only Zs.

        Args:
            operator: The operator to check if it consists only of a single Pauli Z string.

        Returns:
            True if the operator consists of only single a single Pauli Z string (like ``IZZIIZIZ``),
            and False otherwise.
        """
        pauli_z = node.op.operator.paulis[0].z
        qubits = [dag.find_bit(node.qargs[i]).index for i in range(len(node.qargs)) if i in np.nonzero(pauli_z)[0]] 
        return len(node.op.operator.paulis) == 1 and sum(node.op.operator.paulis[0].x) == 0 and sum(node.op.operator.paulis[0].z) > 1 \
            and any(
                np.all(np.linalg.matrix_power(
                    np.array([[self._coupling_map.distance(qubit1, qubit2) <= 1 for qubit2 in qubits] for qubit1 in qubits]), 
                    len(qubits)
                ), axis=0)
            )
    
    
    def _decompose(self, dag: DAGCircuit, node: DAGOpNode) -> DAGCircuit:
        """Decompose the SparsePauliOp into CX and RZ on a coupling map.

        Args:
            dag: The dag needed to get access to qubits.
            node: The operator with the Pauli term we need to apply.

        Returns:
            A dag made of two-qubit :class:`.PauliEvolutionGate`.
        """
        sub_dag = dag.copy_empty_like()
        
        pauli_z = node.op.operator.paulis[0].z
        qubits = [dag.find_bit(node.qargs[i]).index for i in range(len(node.qargs)) if i in np.nonzero(pauli_z)[0]] 
        
        qubits_copy = list(qubits.copy())
        cx_gates = []
        while len(qubits_copy) > 2:
            applied = False
            for qubit in qubits_copy:
                neighbours = [self._coupling_map.distance(qubit, qubit2) == 1 for qubit2 in qubits_copy]
                num_neighbours = sum(neighbours)
                if num_neighbours == 0:
                    logger.info(node.label)
                    logger.info(f'Initial qubits: {qubits}')
                    logger.info(f'Current qubits: {qubits_copy}')
                    logger.info(f'Qubit: {qubit}')
                    logger.info(f'Neighbours: { [[self._coupling_map.distance(qubit1, qubit2) == 1 for qubit2 in qubits] for qubit1 in qubits]}')
                    raise Exception('Disconnected qubit in decomposition')
                if num_neighbours == 1:
                    neighbour = qubits_copy[neighbours.index(True)]
                    sub_dag.apply_operation_back(CXGate(), [dag.qubits[qubit], dag.qubits[neighbour]])
                    cx_gates.append((qubit, neighbour))
                    qubits_copy.remove(qubit)
                    applied = True
                    break
            if not applied:
                logger.warning('No single neighbour qubit found. Apply a random cx to break loops')
                qubit = qubits_copy[0]
                neighbours = [self._coupling_map.distance(qubit, qubit2) == 1 for qubit2 in qubits_copy]
                neighbour = qubits_copy[neighbours.index(True)]
                sub_dag.apply_operation_back(CXGate(), [dag.qubits[qubit], dag.qubits[neighbour]])
                cx_gates.append((qubit, neighbour))
                qubits_copy.remove(qubit)
            
        sub_dag.apply_operation_back(RZZGate(2 * node.op.time), [dag.qubits[qubits_copy[0]], dag.qubits[qubits_copy[1]]])
        for gate in cx_gates[::-1]:
            sub_dag.apply_operation_back(CXGate(), [dag.qubits[gate[0]], dag.qubits[gate[1]]])

        return sub_dag



class FindCommutingPauliEvolutionsMulti(TransformationPass):
    """Finds :class:`.PauliEvolutionGate`s where the operators, that are evolved, all commute."""

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Check for :class:`.PauliEvolutionGate`s where the summands all commute.

        Args:
            The DAG circuit in which to look for the commuting evolutions.

        Returns:
            The dag in which :class:`.PauliEvolutionGate`s made of commuting Paulis
            have been replaced with :class:`.CommutingBlocks`` gate instructions. These gates
            contain nodes of :class:`.PauliEvolutionGate`s.
        """

        for node in dag.op_nodes():
            if isinstance(node.op, PauliEvolutionGate):
                operator: SparsePauliOp = node.op.operator
                if self.single_qubit_terms_only(operator):
                    continue

                if self.summands_commute(operator):
                    sub_dag = self._decompose(dag, node)

                    block_op = CommutingBlock(set(sub_dag.op_nodes()))
                    wire_order = {
                        wire: idx
                        for idx, wire in enumerate(sub_dag.qubits)
                        if wire not in sub_dag.idle_wires()
                    }
                    try:
                        dag.replace_block_with_op([node], block_op, wire_order)
                        # dag.replace_block_with_op([node], CommutingBlock([]), wire_order)
                    except Exception as e:
                        logger.info(e)
                        logger.info(node.num_qubits)
                        logger.info(node.qargs)
                        logger.info(node.op)
                        logger.info(node.op.num_qubits)
                        raise e

        return dag

    @staticmethod
    def single_qubit_terms_only(operator: SparsePauliOp) -> bool:
        """Determine if the Paulis are made of single qubit terms only.

        Args:
            operator: The operator to check if it consists only of single qubit terms.

        Returns:
            True if the operator consists of only single qubit terms (like ``IIX + IZI``),
            and False otherwise.
        """

        for pauli in operator.paulis:
            if sum(np.logical_or(pauli.x, pauli.z)) > 1:
                return False

        return True

    @staticmethod
    def summands_commute(operator: SparsePauliOp) -> bool:
        """Check if all summands in the evolved operator commute.

        Args:
            operator: The operator to check if all its summands commute.

        Returns:
            True if all summands commute, False otherwise.
        """
        # get a list of summands that commute
        commuting_subparts = operator.paulis.group_qubit_wise_commuting()

        # if all commute we only have one summand!
        return len(commuting_subparts) == 1

    @staticmethod
    def _pauli_to_edge(pauli: Pauli) -> Tuple[int, ...]:
        """Convert a pauli to an edge.

        Args:
            pauli: A pauli that is converted to a string to find out where non-identity
                Paulis are.

        Returns:
            A tuple representing where the Paulis are. For example, the Pauli "IZIZ" will
            return (0, 2) since virtual qubits 0 and 2 interact.

        Raises:
            QiskitError: If the pauli does not exactly have two non-identity terms.
        """
        edge = tuple(np.logical_or(pauli.x, pauli.z).nonzero()[0])

        # if len(edge) != 2:
        #     raise QiskitError(f"{pauli} does not have length two.")

        return edge

    def _decompose(self, dag: DAGCircuit, node: DAGOpNode) -> DAGCircuit:
        """Decompose the SparsePauliOp into local-qubit.

        Args:
            dag: The dag needed to get access to qubits.
            op: The operator with all the Pauli terms we need to apply.

        Returns:
            A dag made of two-qubit :class:`.PauliEvolutionGate`.
        """
        sub_dag = dag.copy_empty_like()
        op: PauliEvolutionGate = node.op
        required_paulis = {
            self._pauli_to_edge(pauli): (pauli, coeff)
            for pauli, coeff in zip(op.operator.paulis, op.operator.coeffs)
        }

        for edge, (pauli, coeff) in required_paulis.items():
            
            qubits = [node.qargs[edge[i]] for i in range(len(edge))]

            simple_pauli = Pauli(pauli.to_label().replace("I", ""))

            pauli = PauliEvolutionGate(simple_pauli, op.time * np.real(coeff))
            sub_dag.apply_operation_back(pauli, qubits)
        
        return sub_dag



class ExtendedSwapStrategy(SwapStrategy):
    def __init__(
        self, coupling_map: CouplingMap, swap_layers: tuple[tuple[tuple[int, int], ...], ...], type: str="custom"
    ) -> None:
        self._distances = {}
        self._distance_tensors: dict[int, np.ndarray] = {}
        self._type = type
        super().__init__(coupling_map, swap_layers)
      
      
    @classmethod  
    def from_line(cls, line: list[int], num_swap_layers: int | None = None) -> ExtendedSwapStrategy:
        if len(line) < 2:
            raise ValueError(f"The line cannot have less than two elements, but is {line}")

        if num_swap_layers is None:
            num_swap_layers = len(line) - 2

        elif num_swap_layers < 0:
            raise ValueError(f"Negative number {num_swap_layers} passed for number of swap layers.")

        swap_layer0 = tuple((line[i], line[i + 1]) for i in range(0, len(line) - 1, 2))
        swap_layer1 = tuple((line[i], line[i + 1]) for i in range(1, len(line) - 1, 2))
        # Maybe we want even less structure to increase the mixing?
        def random_shuffle():
            num_swaps = rng.integers(len(line))
            choices = rng.choice(range(len(line)-1), num_swaps, replace=False)
            choices = sorted(list(choices))
            to_delete = []
            for i in range(len(choices) - 1):
                if choices[i] in to_delete:
                    continue
                elif choices[i+1] == choices[i] + 1:
                    to_delete.append(choices[i+1])
            for i in to_delete:
                choices.remove(i)
            return tuple((line[i], line[i + 1]) for i in choices)

        # reshuffle_layer = tuple((line[i], line[i + 1]) for i in range(0, len(line) - 1, 4))

        base_layers = [swap_layer0, swap_layer1]

        rate_of_random = len(line) // 4
        swap_layers = reduce(
            list.__add__,
            [[base_layers[(i+ j*rate_of_random) % 2] for i in range(rate_of_random)] + [random_shuffle()] for j in range(num_swap_layers // rate_of_random)],
            []
        ) + [base_layers[(i + (num_swap_layers // rate_of_random) * rate_of_random) % 2] for i in range(num_swap_layers % rate_of_random)]


        couplings = []
        for idx in range(len(line) - 1):
            couplings.append((line[idx], line[idx + 1]))
            couplings.append((line[idx + 1], line[idx]))

        return cls(coupling_map=CouplingMap(couplings), swap_layers=tuple(swap_layers), type="line")


    @classmethod  
    def from_grid(cls, rows: int, cols: int) -> ExtendedSwapStrategy:

        qubits = [(row, col) for row in range(rows) for col in range(cols)]
        mapping = {qubit: idx for idx, qubit in enumerate(qubits)}

        swap_layer0 = tuple(
            (mapping[(row, col)], mapping[(row, col+1)]) 
            for row in range(rows)
            for col in range(row % 2, cols - 1, 2)
        )
        swap_layer1 = tuple(
            (mapping[(row, col)], mapping[(row, col+1)]) 
            for row in range(rows)
            for col in range((row+1) % 2, cols - 1, 2)
        )
        swap_layer2 = tuple(
            (mapping[(row, col)], mapping[(row+1, col)]) 
            for col in range(cols)
            for row in range(0, rows - 1, 2)
        )
        swap_layer3 = tuple(
            (mapping[(row, col)], mapping[(row+1, col)]) 
            for col in range(cols)
            for row in range(1, rows - 1, 2)
        )

        base_layers = [swap_layer0, swap_layer1]
        swap_layers = reduce(
            list.__add__,
            [
                [base_layers[i % 2] for i in range(cols-1)] + [swap_layer2, swap_layer3] for _ in range(int(np.ceil(rows/2)))
            ],
            []
        )
        
        couplings = []
        for row, col in product(range(rows-1), range(cols-1)):
            new_couplings = [
                (mapping[(row, col)], mapping[(row+1, col)]),
                (mapping[(row, col)], mapping[(row, col+1)])
            ]
            couplings.extend(new_couplings + [c[::-1] for c in new_couplings])
        for col in range(cols - 1):
            new_couplings = [
                (mapping[(rows-1, col)], mapping[(rows-1, col+1)])
            ]
            couplings.extend(new_couplings + [c[::-1] for c in new_couplings])
        for row in range(rows - 1):
            new_couplings = [
                (mapping[(row, cols-1)], mapping[(row+1, cols-1)])
            ]
            couplings.extend(new_couplings + [c[::-1] for c in new_couplings])

        return cls(coupling_map=CouplingMap(couplings), swap_layers=tuple(swap_layers), type="grid")

    
    @classmethod  
    def from_heavy_hex(cls, rows: int, cols: int) -> ExtendedSwapStrategy:
        
        hex = nx.hexagonal_lattice_graph(cols, rows)
        coupling_graph = nx.Graph()
        counter = 0
        mapping = {}
        index_to_name_mapping = {}

        a_nodes = []
        b_nodes = []

        for node in hex.nodes:
            coupling_graph.add_node(counter)
            mapping[node] = counter
            index_to_name_mapping[counter] = node
            counter += 1
        for edge in hex.edges:
            coupling_graph.add_node(counter)
            
            if edge[0][0] != edge[1][0]:
                if (edge[0][0] == 0 and edge[0][1] == 2*cols): # or (edge[0][0] == rows-1 and edge[0][1] == 2*cols) : leave the last one out
                    pass
                # Even row
                elif edge[0][0] % 2 == 0 and edge[0][1] % 4 == 2 and not edge[0][1] == 0:
                    a_nodes.append(edge)
                # Even row
                elif edge[0][0] % 2 == 0 and edge[0][1] % 4 == 0 and not edge[0][1] == 0:
                    b_nodes.append(edge)
                # Odd row
                elif edge[0][0] % 2 == 1 and edge[0][1] % 4 == 1 and not edge[0][1] == 2*cols+1:
                    a_nodes.append(edge)
                # Odd row
                elif edge[0][0] % 2 == 1 and edge[0][1] % 4 == 3 and not edge[0][1] == 2*cols+1:
                    b_nodes.append(edge)
                    
            
            mapping[edge] = counter
            mapping[edge[::-1]] = counter
            index_to_name_mapping[counter] = edge
            counter += 1
            
            
        for node in hex.nodes:
            for edge in hex.edges(node):
                coupling_graph.add_edge(mapping[node], mapping[edge])
                
        coupling_map = CouplingMap(
            list(coupling_graph.edges) + [e[::-1] for e in coupling_graph.edges]
        )


        line_node_counters = [mapping[((0, 2*cols), (1, 2*cols))]]

        for col_idx in range(2*cols,0,-1):
            line_node_counters.append(mapping[(0, col_idx)])
            line_node_counters.append(mapping[((0, col_idx),(0, col_idx-1))])
        line_node_counters.append(mapping[(0, 0)])
        line_node_counters.append(mapping[((0, 0), (1, 0))])

        for row_idx in range(1, rows):
            if row_idx % 2 == 1:
                for col_idx in range(2*cols+1):
                    line_node_counters.append(mapping[(row_idx, col_idx)])
                    line_node_counters.append(mapping[((row_idx, col_idx), (row_idx, col_idx+1))])
                line_node_counters.append(mapping[(row_idx, 2*cols+1)])
                line_node_counters.append(mapping[((row_idx, 2*cols+1), (row_idx+1, 2*cols+1))])
            else:
                for col_idx in range(2*cols+1,0,-1):
                    line_node_counters.append(mapping[(row_idx, col_idx)])
                    line_node_counters.append(mapping[((row_idx, col_idx), (row_idx, col_idx-1))])
                line_node_counters.append(mapping[(row_idx, 0)])
                line_node_counters.append(mapping[((row_idx, 0), (row_idx+1, 0))])
                
        if rows % 2 == 0:
            for col_idx in range(2*cols+1,1,-1):
                line_node_counters.append(mapping[(rows, col_idx)])
                line_node_counters.append(mapping[((rows, col_idx), (rows, col_idx-1))])
            line_node_counters.append(mapping[(rows, 1)])
            # line_node_counters.append(mapping[((rows, 1), (rows-1, 1))])  : leave the last one out   
        else:
            for col_idx in range(2*cols):
                line_node_counters.append(mapping[(rows, col_idx)])
                line_node_counters.append(mapping[((rows, col_idx), (rows, col_idx+1))])
            line_node_counters.append(mapping[(rows, 2*cols)])
            # line_node_counters.append(mapping[((rows, 2*rows), (rows-1, 2*rows))])      : leave the last one out      

        swap_1 = tuple((line_node_counters[i], line_node_counters[i + 1]) for i in range(0, len(line_node_counters) - 1, 2))
        swap_2 = tuple((line_node_counters[i], line_node_counters[i + 1]) for i in range(1, len(line_node_counters) - 1, 2))
        swap_3 = tuple((mapping[a_node], mapping[a_node[0]]) for a_node in a_nodes)
        swap_4 = tuple((mapping[b_node], mapping[b_node[0]]) for b_node in b_nodes)
        l = len(line_node_counters)
        k = int(l/4 - (l/4 % 8) + 10)
        
        swap_layers = []
        for i in range(k - 7):
            swap_layers.append(swap_1 if i % 2 == 0 else swap_2)
        swap_layers.append(swap_4)
        for i in range(7):
            swap_layers.append(swap_2 if i % 2 == 0 else swap_1)
        swap_layers.append(swap_3)

        swap_layers = tuple(swap_layers * 5)

        return cls(coupling_map=coupling_map, swap_layers=tuple(swap_layers), type="heavy_hex")
    
    
    def distance_nodes(self, nodes: tuple) -> int | float:
        nodes = tuple(sorted(nodes))
        distance = self._distances.get(nodes, None)
        if distance is not None:
            return distance
        
        if len(nodes) < 2:
            return 0
        
        for i in range(len(self._swap_layers) + 1):
            cmap = self.swapped_coupling_map(i)
            distance_matrix: npt.NDArray[np.float64] = cmap.distance_matrix
            sub_distance = distance_matrix[nodes, :][:, nodes]
            if (
                np.max(sub_distance) <= len(nodes) - 1
                and
                np.any(
                    np.all(np.linalg.matrix_power(sub_distance <= 1, len(nodes)) > 0, axis=1)
                ) # If nodes form a connected subgraph
            ):
                self._distances[nodes] = i
                return i
        return math.inf
    
    
    def all_connected_subgraphs(self, layer: int, order: int):
        cmap = self.swapped_coupling_map(layer)
        g = [set(cmap.neighbors(q)) for q in cmap.physical_qubits]
        def _recurse(t: tuple, possible: set[int], excluded: set[int]):
            if len(t) == order:
                yield t
            else:
                excluded = set(excluded)
                for i in possible.difference(excluded):
                    new_t = (*t, i)
                    new_possible = possible | g[i]
                    excluded.add(i)
                    yield from _recurse(new_t, new_possible, excluded)
        excluded = set()
        for (i, possible) in enumerate(g):
            excluded.add(i)
            yield from _recurse((i,), possible, excluded)
    
    
    def distance_tensor(self, order) -> np.ndarray:
        if order == 2:
            return self.distance_matrix
        
        dt = self._distance_tensors.get(order, None)
        if dt is not None:
            return dt
        
        try:
            with open(f'/lustre/scratch127/qpg/jc59/hubo/swap_strategy_{self._type}_distance_qubits_{self._num_vertices}_order_{order}.pkl', 'rb') as f:
                dt = pickle.load(f)
                logger.info("Loaded data")
                return dt
        except FileNotFoundError:
            pass
        
        
        distance_tensor = np.full([self._num_vertices]*order, -1)        
        for i in range(len(self._swap_layers) + 1):
            subgraphs = self.all_connected_subgraphs(i, order)
            for subgraph in subgraphs:
                subgraph = tuple(sorted(subgraph))
                if distance_tensor[subgraph] == -1:
                    self._distances[subgraph] = i
                    for perm in permutations(subgraph, len(subgraph)):
                        distance_tensor[perm] = i

                    
        self._distance_tensors[order] = distance_tensor
        
        with open(f'/lustre/scratch127/qpg/jc59/hubo/swap_strategy_{self._type}_distance_qubits_{self._num_vertices}_order_{order}.pkl', 'wb') as f:
            pickle.dump(distance_tensor, f)
        
        return distance_tensor
                    


class CommutingGateRouter(TransformationPass):
    def __init__(
        self,
        swap_strategy: ExtendedSwapStrategy | None = None,
        edge_coloring: dict[tuple[int, int], int] | None = None,
        max_layers: int | None = None,
        perform_extra_swaps: bool = True
    ) -> None:
        r"""
        Args:
            swap_strategy: An instance of a :class:`.SwapStrategy` that holds the swap layers
                that are used, and the order in which to apply them, to map the instruction to
                the hardware. If this field is not given, it should be contained in the
                property set of the pass. This allows other passes to determine the most
                appropriate swap strategy at run-time.
            edge_coloring: An optional edge coloring of the coupling map (I.e. no two edges that
                share a node have the same color). If the edge coloring is given then the commuting
                gates that can be simultaneously applied given the current qubit permutation are
                grouped according to the edge coloring and applied according to this edge
                coloring. Here, a color is an int which is used as the index to define and
                access the groups of commuting gates that can be applied simultaneously.
                If the edge coloring is not given then the sets will be built-up using a
                greedy algorithm. The edge coloring is useful to position gates such as
                ``RZZGate``\s next to swap gates to exploit CX cancellations.
        """
        super().__init__()
        self._swap_strategy = swap_strategy
        self._bit_indices: dict[Qubit, int] | None = None
        self._edge_coloring = edge_coloring
        if max_layers is None and swap_strategy is not None:
            max_layers = len(swap_strategy)
        self._max_layers = max_layers
        self._perform_extra_swaps = perform_extra_swaps

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the pass by decomposing the nodes it applies on.

        Args:
            dag: The dag to which we will add swaps.

        Returns:
            A dag where swaps have been added for the intended gate type.

        Raises:
            TranspilerError: If the swap strategy was not given at init time and there is
                no swap strategy in the property set.
            TranspilerError: If the quantum circuit contains more than one qubit register.
            TranspilerError: If there are qubits that are not contained in the quantum register.
        """
        if self._swap_strategy is None:
            swap_strategy = self.property_set["swap_strategy"]

            if swap_strategy is None:
                raise TranspilerError("No swap strategy given at init or in the property set.")
            elif self._max_layers is None:
                self._max_layers = len(swap_strategy)
        else:
            swap_strategy = self._swap_strategy

        if len(dag.qregs) != 1:
            raise TranspilerError(
                f"{self.__class__.__name__} runs on circuits with one quantum register."
            )

        if len(dag.qubits) != next(iter(dag.qregs.values())).size:
            raise TranspilerError("Circuit has qubits not contained in the qubit register.")

        # Fix output permutation -- copied from ElidePermutations
        input_qubit_mapping = {qubit: index for index, qubit in enumerate(dag.qubits)}
        self.property_set["original_layout"] = Layout(input_qubit_mapping)
        if self.property_set["original_qubit_indices"] is None:
            self.property_set["original_qubit_indices"] = input_qubit_mapping

        new_dag = dag.copy_empty_like()
        current_layout = Layout.generate_trivial_layout(*dag.qregs.values())

        # Used to keep track of nodes that do not decompose using swap strategies.
        accumulator = new_dag.copy_empty_like()
        cannot_implement = []
        
        for node in dag.topological_op_nodes():
            if isinstance(node.op, CommutingBlock):

                # Check that the swap strategy creates enough connectivity for the node.
                cannot_implement.extend(self._check_edges(dag, node, swap_strategy))

                # Compose any accumulated non-swap strategy gates to the dag
                accumulator = self._compose_non_swap_nodes(accumulator, current_layout, new_dag)

                # Decompose the swap-strategy node and add to the dag.
                new_dag.compose(self.swap_decompose(dag, node, current_layout, swap_strategy))
            else:
                logger.info('Not commuting block')
                accumulator.apply_operation_back(node.op, node.qargs, node.cargs)

        # TODO: find the best point to implement them, rather than dumping at the end i.e. the time when minimum distance for the ops
        for sub_node in cannot_implement:
            accumulator.apply_operation_back(sub_node.op, sub_node.qargs, sub_node.cargs)
        
        logger.info(f'Gates we cannot directly implement: {len(cannot_implement)}')
        logger.info([tuple([dag.find_bit(sub_node.qargs[i]).index for i in range(len(sub_node.qargs))]) for sub_node in cannot_implement])
        
        if self._perform_extra_swaps:
            self._compose_non_swap_nodes(accumulator, current_layout, new_dag)
        else:
            logger.info('Not implementing those gates')

        self.property_set["virtual_permutation_layout"] = current_layout

        return new_dag

    def _compose_non_swap_nodes(
        self, accumulator: DAGCircuit, layout: Layout, new_dag: DAGCircuit
    ) -> DAGCircuit:
        """Add all the non-swap strategy nodes that we have accumulated up to now.

        This method also resets the node accumulator to an empty dag.

        Args:
            layout: The current layout that keeps track of the swaps.
            new_dag: The new dag that we are building up.
            accumulator: A DAG to keep track of nodes that do not decompose
                using swap strategies.

        Returns:
            A new accumulator with the same registers as ``new_dag``.
        """
        # Add all the non-swap strategy nodes that we have accumulated up to now.
        order = layout.reorder_bits(new_dag.qubits)
        order_bits: list[int | None] = [None] * len(layout)
        for idx, val in enumerate(order):
            order_bits[val] = idx

        new_dag.compose(accumulator, qubits=order_bits)

        # Re-initialize the node accumulator
        return new_dag.copy_empty_like()

    def _position_in_cmap(self, dag: DAGCircuit, indices: tuple[int], layout: Layout) -> tuple[int, ...]:
        """A helper function to track the movement of virtual qubits through the swaps.

        Args:
            indices: The index of decision variables (i.e. virtual qubit).
            layout: The current layout that takes into account previous swap gates.

        Returns:
            The position in the coupling map of the virtual qubits as a tuple.
        """
        bits = tuple([dag.find_bit(layout.get_physical_bits()[i]).index for i in indices])

        return bits

    def _build_sub_layers(
        self, current_layer: dict[tuple[int, int], Gate]
    ) -> list[dict[tuple[int, int], Gate]]:
        """A helper method to build-up sets of gates to simultaneously apply.

        This is done with an edge coloring if the ``edge_coloring`` init argument was given or with
        a greedy algorithm if not. With an edge coloring all gates on edges with the same color
        will be applied simultaneously. These sublayers are applied in the order of their color,
        which is an int, in increasing color order.

        Args:
            current_layer: All gates in the current layer can be applied given the qubit ordering
                of the current layout. However, not all gates in the current layer can be applied
                simultaneously. This function creates sub-layers by building up sub-layers
                of gates. All gates in a sub-layer can simultaneously be applied given the coupling
                map and current qubit configuration.

        Returns:
             A list of gate dicts that can be applied. The gates a position 0 are applied first.
             A gate dict has the qubit tuple as key and the gate to apply as value.
        """
        if self._edge_coloring is not None:
            return self._edge_coloring_build_sub_layers(current_layer)
        else:
            return self._greedy_build_sub_layers_2(current_layer)

    def _edge_coloring_build_sub_layers(
        self, current_layer: dict[tuple[int, int], Gate]
    ) -> list[dict[tuple[int, int], Gate]]:
        """The edge coloring method of building sub-layers of commuting gates."""
        sub_layers: list[dict[tuple[int, int], Gate]] = [
            {} for _ in set(self._edge_coloring.values())
        ]
        greedy_gates = {}
        for edge, gate in current_layer.items():
            if len(edge) == 2:
                color = self._edge_coloring[tuple(sorted(edge))]
                sub_layers[color][edge] = gate
            else:
                greedy_gates[edge] = gate
        greed_sub_layers = self._greedy_build_sub_layers_2(greedy_gates)
        sub_layers.extend(greed_sub_layers)
        return sub_layers
    
    
    @staticmethod
    def _greedy_build_sub_layers_2(
        current_layer: dict[tuple[int, int], Gate]
    ) -> list[dict[tuple[int, int], Gate]]:
        """The greedy method of building sub-layers of commuting gates.
        We have multi-qubit Z rotations, which need to be decomposed into multiple layers, throughout all of which each qubit is blocked out.
        
        """
        sub_layers = []
        current_sub_layer_index = 0
        while len(current_layer) > 0:
            current_sub_layer, remaining_gates = {}, {}
            blocked_vertices: dict[int, set[tuple]] = {}

            for edge, evo_gate in sorted(current_layer.items(), key=lambda e: -len(e[0])):
                if blocked_vertices.get(current_sub_layer_index, set()).isdisjoint(edge):
                    current_sub_layer[edge] = evo_gate

                    # A vertex becomes blocked once a gate is applied to it.
                    for i in range(max(0, 2 * (len(edge) - 2)) + 1):
                        bv = blocked_vertices.get(current_sub_layer_index + i, set())
                        bv = bv.union(edge)
                        blocked_vertices[current_sub_layer_index + i] = bv
                else:
                    remaining_gates[edge] = evo_gate

            current_layer = remaining_gates
            sub_layers.append(current_sub_layer)
            current_sub_layer_index += 1

        return sub_layers
    

    @staticmethod
    def _greedy_build_sub_layers(
        current_layer: dict[tuple[int, int], Gate]
    ) -> list[dict[tuple[int, int], Gate]]:
        """The greedy method of building sub-layers of commuting gates.
        We have multi-qubit Z rotations, which need to be decomposed into multiple layers, throughout all of which each qubit is blocked out.
        
        """
        sub_layers = []
        while len(current_layer) > 0:
            current_sub_layer, remaining_gates = {}, {}
            blocked_vertices: set[tuple] = set()

            for edge, evo_gate in sorted(current_layer.items(), key=lambda e: -len(e[0])):
                if blocked_vertices.isdisjoint(edge):
                    current_sub_layer[edge] = evo_gate

                    # A vertex becomes blocked once a gate is applied to it.
                    blocked_vertices = blocked_vertices.union(edge)
                else:
                    remaining_gates[edge] = evo_gate

            current_layer = remaining_gates
            sub_layers.append(current_sub_layer)

        return sub_layers

    def swap_decompose(
        self, dag: DAGCircuit, node: DAGOpNode, current_layout: Layout, swap_strategy: ExtendedSwapStrategy
    ) -> DAGCircuit:
        """Take an instance of :class:`.Commuting2qBlock` and map it to the coupling map.

        The mapping is done with the swap strategy.

        Args:
            dag: The dag which contains the :class:`.Commuting2qBlock` we route.
            node: A node whose operation is a :class:`.Commuting2qBlock`.
            current_layout: The layout before the swaps are applied. This function will
                modify the layout so that subsequent gates can be properly composed on the dag.
            swap_strategy: The swap strategy used to decompose the node.

        Returns:
            A dag that is compatible with the coupling map where swap gates have been added
            to map the gates in the :class:`.Commuting2qBlock` to the hardware.
        """
        trivial_layout = Layout.generate_trivial_layout(*dag.qregs.values())
        gate_layers = self._make_op_layers(dag, node.op, current_layout, swap_strategy)

        # Iterate over and apply gate layers
        max_distance = max(gate_layers.keys())
        logger.info(f'Max layers needed to apply swap decompose: {max_distance}')

        circuit_with_swap = QuantumCircuit(len(dag.qubits))

        for i in range(max_distance + 1):
            # Get current layer and replace the problem indices j,k by the corresponding
            # positions in the coupling map. The current layer corresponds
            # to all the gates that can be applied at the ith swap layer.
            current_layer = {}
            for indices, local_gate in gate_layers.get(i, {}).items():
                current_layer[self._position_in_cmap(dag, indices, current_layout)] = local_gate

            # Not all gates that are applied at the ith swap layer can be applied at the same
            # time. We therefore greedily build sub-layers.
            # logger.info(f'Layer {i}. Gates in layer: {current_layer.keys()}')
            # logger.info(f'Unmapped gates in layer: {[indices for indices, _ in gate_layers.get(i, {}).items()]}')
            sub_layers = self._build_sub_layers(current_layer)
            # logger.info(f'Layer {i}. Sub-layers: {len(sub_layers)}. Max interaction size: {max([len(key) for key in current_layer.keys()]) if len(current_layer.keys()) > 0 else 0}')

            # Apply sub-layers
            for sublayer in sub_layers:
                for edge, local_gate in sublayer.items():
                    circuit_with_swap.append(local_gate, edge)

            # Apply SWAP gates
            if i < max_distance:
                for swap in swap_strategy.swap_layer(i):
                    try:
                        (j, k) = [trivial_layout.get_physical_bits()[vertex] for vertex in swap]
                    except KeyError:
                        logger.info(swap)
                        logger.info(trivial_layout.get_physical_bits())
                        raise KeyError()

                    circuit_with_swap.swap(j, k)
                    current_layout.swap(j, k)

        return circuit_to_dag(circuit_with_swap)

    def _make_op_layers(
        self, dag: DAGCircuit, op: CommutingBlock, layout: Layout, swap_strategy: ExtendedSwapStrategy
    ) -> dict[int, dict[tuple, Gate]]:
        """Creates layers of two-qubit gates based on the distance in the swap strategy."""

        gate_layers: dict[int, dict[tuple, Gate]] = defaultdict(dict)

        for node in op.node_block:
            edge = tuple([dag.find_bit(node.qargs[i]).index for i in range(len(node.qargs))])

            v_bits = layout.get_virtual_bits()        
            bits = tuple([v_bits[dag.qubits[edge[i]]] for i in range(len(edge))])

            distance = swap_strategy.distance_nodes(bits)
            
            if distance <= self._max_layers:
                gate_layers[distance][edge] = node.op

        return gate_layers

    def _check_edges(self, dag: DAGCircuit, node: DAGOpNode, swap_strategy: ExtendedSwapStrategy):
        """Check if the swap strategy can create the required connectivity.

        Args:
            node: The dag node for which to check if the swap strategy provides enough connectivity.
            swap_strategy: The swap strategy that is being used.

        Raises:
            TranspilerError: If there is an edge that the swap strategy cannot accommodate
                and if the pass has been configured to raise on such issues.
        """
        cannot_implement = []
        for sub_node in node.op:
            bits = tuple([dag.find_bit(sub_node.qargs[i]).index for i in range(len(sub_node.qargs))])
            distance = swap_strategy.distance_nodes(bits)
            if distance > self._max_layers:
                cannot_implement.append(sub_node)
                # raise TranspilerError(
                #     f"{swap_strategy} cannot implement operator on {bits}."
                # )
        return cannot_implement
            
