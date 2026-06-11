"""Qubit routing strategies for the QAOA cost layer.

This module provides ``QUBOSwapStrategy``, a subclass of Qiskit's
``SwapStrategy``, and factory class methods for common hardware topologies.
The swap strategy encodes *which pairs of qubits are swapped at each SWAP
layer*, controlling how quickly all pairs of QUBO variables come into
proximity so that their ZZ interaction can be applied.

For a dense QUBO (every pair of variables interacts) the routing challenge is
to minimise the 2-qubit gate depth.  Different hardware topologies call for
different swap schedules:

* **All-to-all** (``from_all_to_all``): No SWAPs needed; all pairs are already
  adjacent.  Used for small simulators or ideal backends.
* **Line** (``from_line``): Alternating odd/even SWAP layers on a 1-D chain.
  Brings every pair into contact in O(n) layers.
* **Grid** (``from_grid``): Row-then-column SWAP layers on a 2-D grid.
  Interleaves horizontal and vertical passes.
* **Heavy-hex** (``from_heavy_hex``): Designed for IBM heavy-hexagon lattices.
  Uses ``get_longest_line_swaps`` to find a Hamiltonian path through the
  combined qubit-and-coupler graph and schedules SWAPs along that path.
"""
from __future__ import annotations

import pickle
import networkx as nx
import numpy as np
import numpy.typing as npt
from itertools import permutations, product
from functools import reduce

from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import SwapStrategy
from qiskit.transpiler.coupling import CouplingMap


from qiskit_qaoa.utils.logging import get_logger

logger = get_logger(__name__)
rng = np.random.default_rng(seed=1)


def get_longest_line_swaps(mapping: dict, graph: nx.Graph, rows: int, cols: int):
    """Derive an efficient SWAP-layer schedule for a heavy-hexagon lattice.

    Constructs a near-Hamiltonian path through the heavy-hex qubit-coupler
    graph (where both data qubits and coupler nodes are physical qubits) and
    schedules alternating odd/even SWAP layers along that path, followed by
    two additional layers that handle the ``a_nodes`` and ``b_nodes`` — the
    inter-row coupler edges that sit off the main line.  The full schedule is
    repeated five times to expose all qubit pairs.

    Args:
        mapping: Dict mapping heavy-hex node labels (``(row, col)`` for data
            qubits; ``((row, col), (row, col))`` for couplers) to integer
            physical qubit indices.
        graph: The hexagonal-lattice ``nx.Graph`` from ``nx.hexagonal_lattice_graph``
            used to identify inter-row coupler edges.
        rows: Number of hexagonal-lattice rows.
        cols: Number of hexagonal-lattice columns.

    Returns:
        A list of SWAP layers (each layer is a tuple of ``(qubit_i, qubit_j)``
        pairs), sized to route all qubit pairs through the heavy-hex topology.
    """
    a_nodes, b_nodes = [], []

    for edge in graph.edges:            
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
    # Optimal for 2-interactions
    for i in range(k - 7):
        swap_layers.append(swap_1 if i % 2 == 0 else swap_2)
    swap_layers.append(swap_4)
    for i in range(7):
        swap_layers.append(swap_2 if i % 2 == 0 else swap_1)
    swap_layers.append(swap_3)
    
    return swap_layers * 5


class QUBOSwapStrategy(SwapStrategy):
    """Hardware-aware SWAP strategy for routing QAOA cost interactions.

    Extends Qiskit's ``SwapStrategy`` with factory class methods tailored to
    common hardware topologies (all-to-all, line, grid, heavy-hex).  A custom
    strategy is needed because the standard Qiskit strategies do not account
    for the QUBO structure (dense interactions among a subset of qubits) or
    the specific SWAP schedules required to minimise 2-qubit gate depth on
    IBM heavy-hexagon devices.

    The ``_type`` attribute records which topology was used and can be
    inspected downstream for topology-specific optimisations.

    Attributes:
        _distances: Cache for previously computed inter-qubit distances (after
            a given number of SWAP layers).
        _distance_tensors: Cache for distance tensors indexed by SWAP-layer
            count.
        _type: String tag identifying the topology (``"all_to_all"``,
            ``"grid"``, ``"heavy_hex"``, or ``"custom"``).
    """
    def __init__(
        self, coupling_map: CouplingMap, swap_layers: tuple[tuple[tuple[int, int], ...], ...], type: str="custom"
    ) -> None:
        self._distances = {}
        self._distance_tensors: dict[int, np.ndarray] = {}
        self._type = type
        super().__init__(coupling_map, swap_layers)
        
    
    @classmethod
    def from_all_to_all(cls, num_qubits: int) -> QUBOSwapStrategy:
        """Create a strategy for a fully-connected (all-to-all) topology.

        No SWAP layers are needed because every qubit pair is directly
        coupled.  Suitable for small statevector simulators or ideal backends.

        Args:
            num_qubits: Total number of qubits.

        Returns:
            A ``QUBOSwapStrategy`` with an empty SWAP-layer tuple and a
            coupling map containing all directed edges ``(i, j)`` for
            ``i < j``.
        """
        couplings = []
        for i in range(num_qubits - 1):
            for j in range(i+1, num_qubits):
                couplings.append((i, j))
                couplings.append((j, i))

        return cls(coupling_map=CouplingMap(couplings), swap_layers=tuple(), type="all_to_all")
        
      
    @classmethod
    def from_line(cls, line: list[int], num_swap_layers: int | None = None) -> QUBOSwapStrategy:
        """Create a strategy for a 1-D chain of qubits.

        Builds alternating even/odd SWAP layers along ``line``.  Layer 0 swaps
        pairs ``(line[0], line[1]), (line[2], line[3]), …``; layer 1 swaps
        ``(line[1], line[2]), (line[3], line[4]), …``.  The two layers alternate
        for ``num_swap_layers`` total rounds, which defaults to
        ``len(line) - 2`` (enough to route all pairs on a chain).

        Args:
            line: Ordered list of physical qubit indices forming the 1-D chain.
            num_swap_layers: Number of alternating SWAP layers to include in
                the schedule.  Defaults to ``len(line) - 2``.

        Returns:
            A ``QUBOSwapStrategy`` with a bidirectional coupling map and the
            computed alternating SWAP-layer schedule.
        """
        if num_swap_layers is None:
            num_swap_layers = len(line) - 2
        swap_layer0 = tuple((line[i], line[i + 1]) for i in range(0, len(line) - 1, 2))
        swap_layer1 = tuple((line[i], line[i + 1]) for i in range(1, len(line) - 1, 2))

        base_layers = [swap_layer0, swap_layer1]

        swap_layers = tuple(base_layers[i % 2] for i in range(num_swap_layers))

        couplings = []
        for idx in range(len(line) - 1):
            couplings.append((line[idx], line[idx + 1]))
            couplings.append((line[idx + 1], line[idx]))

        return cls(coupling_map=CouplingMap(couplings), swap_layers=tuple(swap_layers))
    

    @classmethod
    def from_grid(cls, rows: int, cols: int) -> QUBOSwapStrategy:
        """Create a strategy for a 2-D rectangular grid of qubits.

        Builds four base SWAP layers (two horizontal, two vertical) and
        combines them into a full schedule that interleaves horizontal column
        sweeps with vertical row sweeps, cycling for ``ceil(rows/2)``
        repetitions.  The horizontal layers alternate between even-column and
        odd-column swaps; the vertical layers alternate between even-row and
        odd-row swaps.

        Args:
            rows: Number of rows in the grid.
            cols: Number of columns in the grid.

        Returns:
            A ``QUBOSwapStrategy`` with a grid coupling map (bidirectional
            nearest-neighbour edges) and the interleaved row/column SWAP-layer
            schedule.
        """
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

        row_layers = [swap_layer0, swap_layer1]
        col_layers = [swap_layer2, swap_layer3]
        full_row_swap_layers = reduce(
            list.__add__,
            [
                [row_layers[i % 2] for i in range(cols-1)] + col_layers for _ in range(int(np.ceil(rows/2)))
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

        return cls(coupling_map=CouplingMap(couplings), swap_layers=tuple(full_row_swap_layers), type="grid")

    
    @classmethod
    def from_heavy_hex(cls, rows: int, cols: int) -> QUBOSwapStrategy:
        """Create a strategy for an IBM heavy-hexagon lattice.

        Constructs the heavy-hex qubit-coupler graph from
        ``nx.hexagonal_lattice_graph``, where each coupler edge becomes an
        additional qubit node.  The coupling map is built from the resulting
        bipartite qubit-coupler connectivity.  SWAP layers are derived from
        ``get_longest_line_swaps``, which finds a near-Hamiltonian path through
        the graph and produces alternating SWAP layers along that path plus
        correction layers for off-path coupler edges.

        Args:
            rows: Number of hexagonal-lattice rows (passed to
                ``nx.hexagonal_lattice_graph`` as the second argument).
            cols: Number of hexagonal-lattice columns (passed as the first
                argument).

        Returns:
            A ``QUBOSwapStrategy`` with the heavy-hex coupling map and the
            SWAP-layer schedule produced by ``get_longest_line_swaps``.
        """
        hex = nx.hexagonal_lattice_graph(cols, rows)
        coupling_graph = nx.Graph()
        counter = 0
        mapping = {}

        for node in hex.nodes:
            coupling_graph.add_node(counter)
            mapping[node] = counter
            counter += 1
        for edge in hex.edges:
            coupling_graph.add_node(counter)
                    
            mapping[edge] = counter
            mapping[edge[::-1]] = counter

            counter += 1
            
            
        for node in hex.nodes:
            for edge in hex.edges(node):
                coupling_graph.add_edge(mapping[node], mapping[edge])
                
        coupling_map = CouplingMap(
            list(coupling_graph.edges) + [e[::-1] for e in coupling_graph.edges]
        )

        row_swap_layers = get_longest_line_swaps(mapping, hex, rows, cols)

        swap_layers = tuple(row_swap_layers)

        return cls(coupling_map=coupling_map, swap_layers=tuple(swap_layers), type="heavy_hex")
    
    
    # def distance_nodes(self, nodes: tuple[int,...], cutoff: int | None = None) -> int:
    #     if cutoff is None:
    #         cutoff = len(self._swap_layers) + 1
    #     nodes = tuple(sorted(nodes))
    #     if np.any([nodes[i] == nodes[i+1] for i in range(len(nodes)-1)]):
    #         return -1
    #     distance = self._distances.get(nodes, None)
    #     if distance is not None:
    #         return distance
        
    #     if len(nodes) < 2:
    #         return 0
        
    #     for i in range(cutoff):
    #         cmap = self.swapped_coupling_map(i)
    #         distance_matrix: npt.NDArray[np.float64] = cmap.distance_matrix
    #         sub_distance = distance_matrix[nodes, :][:, nodes]
    #         if (
    #             np.max(sub_distance) <= len(nodes) - 1
    #             and
    #             np.any(
    #                 np.all(np.linalg.matrix_power(sub_distance <= 1, len(nodes)) > 0, axis=1)
    #             ) # If nodes form a connected subgraph
    #         ):
    #             self._distances[nodes] = i
    #             return i
    #     return -1
    
    
    # def all_connected_subgraphs(self, layer: int, order: int):
    #     cmap = self.swapped_coupling_map(layer)
    #     g = [set(cmap.neighbors(q)) for q in cmap.physical_qubits]
    #     def _recurse(t: tuple, possible: set[int], excluded: set[int]):
    #         if len(t) == order:
    #             yield tuple(sorted(list(t)))
    #         else:
    #             excluded = set(excluded)
    #             for i in possible.difference(excluded):
    #                 new_t = (*t, i)
    #                 new_possible = possible | g[i]
    #                 excluded.add(i)
    #                 yield from _recurse(new_t, new_possible, excluded)
    #     excluded = set()
    #     for (i, possible) in enumerate(g):
    #         excluded.add(i)
    #         yield from _recurse((i,), possible, excluded)
    