"""ExtendedSwapStrategy for QAOA gate routing on different hardware topologies.

A swap strategy is a precomputed sequence of SWAP-gate layers that, when
applied repeatedly, can bring any pair (or tuple) of qubits into adjacency so
that a two-qubit (or multi-qubit) gate can be applied.  ``ExtendedSwapStrategy``
extends Qiskit's ``SwapStrategy`` with:

- Factory class methods for common topologies (line, grid, heavy-hex,
  all-to-all).
- Lazy caching of pairwise and higher-order distance tensors, with on-disk
  persistence for expensive computations.
- ``distance_nodes`` for querying the minimum swap depth needed to make a
  tuple of qubits mutually adjacent (i.e. form a connected subgraph).
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
    # Optimal for 2-interactions; equiv to W = 1 below
    # for i in range(k - 7):
    #     swap_layers.append(swap_1 if i % 2 == 0 else swap_2)
    # swap_layers.append(swap_4)
    # for i in range(7):
    #     swap_layers.append(swap_2 if i % 2 == 0 else swap_1)
    # swap_layers.append(swap_3)
    # swap_layers = swap_layers * 5
    
    W = 3 if (rows > 1 and cols > 1) else 1
    for _ in range(W - 1):
        for i in range((k - 7) // W):
            swap_layers.append(swap_1 if i % 2 == 0 else swap_2)
        swap_layers.append(swap_4)
        for i in range(7 // W):
            swap_layers.append(swap_2 if i % 2 == 0 else swap_1)
        swap_layers.append(swap_3)
        
    for i in range((k - 7) // W + (k - 7) % W):
        swap_layers.append(swap_1 if i % 2 == 0 else swap_2)
    swap_layers.append(swap_4)
    for i in range(7 // W + 7 % W):
        swap_layers.append(swap_2 if i % 2 == 0 else swap_1)
    swap_layers.append(swap_3)
    
    
    swap_layers = swap_layers * 5
    
    return swap_layers


class ExtendedSwapStrategy(SwapStrategy):
    """Swap strategy with distance caching and multi-topology factory constructors.

    Inherits from Qiskit's ``SwapStrategy`` and adds:

    - Per-instance distance caches (``_distances`` for tuple queries,
      ``_distance_tensors`` for full-order tensors).
    - Factory class methods for line, 2-D grid, heavy-hex, and all-to-all
      topologies, each generating the appropriate swap-layer sequence.
    - ``distance_nodes``: minimum swap depth to make a qubit tuple mutually
      adjacent (i.e. form a connected subgraph), with LRU-style cache look-up.
    - ``distance_tensor``: full ``n^order`` tensor of pairwise / higher-order
      distances, computed lazily and persisted to disk.

    Args:
        coupling_map: The physical coupling map of the hardware topology.
        swap_layers: An ordered tuple of swap layers; each layer is a tuple
            of ``(qubit_a, qubit_b)`` pairs to be swapped simultaneously.
        type: A string label for the topology (used in the on-disk cache
            filename).  Defaults to ``"custom"``.
    """

    def __init__(
        self, coupling_map: CouplingMap, swap_layers: tuple[tuple[tuple[int, int], ...], ...], type: str="custom"
    ) -> None:
        self._distances = {}
        self._distance_tensors: dict[int, np.ndarray] = {}
        self._type = type
        super().__init__(coupling_map, swap_layers)
        
    
    @classmethod
    def from_all_to_all(cls, num_qubits: int) -> ExtendedSwapStrategy:
        couplings = []
        for i in range(num_qubits - 1):
            for j in range(i+1, num_qubits):
                couplings.append((i, j))
                couplings.append((j, i))

        return cls(coupling_map=CouplingMap(couplings), swap_layers=tuple(), type="all_to_all")
        
      
      
    @classmethod
    def from_line(cls, line: list[int], num_swap_layers: int | None = None) -> ExtendedSwapStrategy:
        """Construct a swap strategy for a linear qubit chain.

        Alternates between even-indexed and odd-indexed nearest-neighbour swap
        layers, with occasional randomly-generated shuffle layers inserted at
        a rate of ``len(line) // 4`` to increase mixing.

        Args:
            line: Ordered list of physical qubit indices forming the chain.
                Must have at least two elements.
            num_swap_layers: Total number of swap layers to generate.  Defaults
                to ``len(line) - 2``.

        Returns:
            An ``ExtendedSwapStrategy`` with ``type="line"``.

        Raises:
            ValueError: If ``line`` has fewer than 2 elements or if
                ``num_swap_layers`` is negative.
        """
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
        """Construct a swap strategy for a 2-D rectangular qubit grid.

        Generates four base swap layers (two alternating row-direction and two
        alternating column-direction) and combines them into a full sequence
        that cycles through both dimensions.

        Args:
            rows: Number of rows in the grid.
            cols: Number of columns in the grid.

        Returns:
            An ``ExtendedSwapStrategy`` with ``type="grid"``.
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
            for row in range(col % 2, rows - 1, 2)
        )
        swap_layer3 = tuple(
            (mapping[(row, col)], mapping[(row+1, col)]) 
            for col in range(cols)
            for row in range((col+1) % 2, rows - 1, 2)
        )

        row_layers = [swap_layer0, swap_layer1]
        col_layers = [swap_layer2, swap_layer3]
        full_row_swap_layers = reduce(
            list.__add__,
            [
                [row_layers[i % 2] for i in range(cols-1)] + col_layers for _ in range(int(np.ceil(rows/2)) - 1)
            ] + [
                [row_layers[i % 2] for i in range(cols-1)]
            ],
            []
        )
        full_col_swap_layers = reduce(
            list.__add__,
            [
                [col_layers[i % 2] for i in range(rows-1)] + row_layers for _ in range(int(np.ceil(cols/2)) - 1)
            ] + [
                [col_layers[i % 2] for i in range(rows-1)]
            ],
            []
        )
        swap_layers = (full_row_swap_layers + full_col_swap_layers) * (rows+cols)
        
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
        """Construct a swap strategy for a heavy-hex IBM quantum processor topology.

        Builds the heavy-hex coupling graph using NetworkX's hexagonal lattice
        graph, inserting auxiliary qubits on each edge (as physical qubits).
        The swap layers follow the longest-line traversal of the heavy-hex
        lattice produced by ``get_longest_line_swaps``.

        Args:
            rows: Number of hexagonal rows.
            cols: Number of hexagonal columns.

        Returns:
            An ``ExtendedSwapStrategy`` with ``type="heavy_hex"``.
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
    
    
    def distance_nodes(self, nodes: tuple[int,...], cutoff: int | None = None) -> int:
        """Return the minimum swap depth needed to make a qubit tuple mutually adjacent.

        A tuple of qubits is "implementable" at swap depth ``d`` if, after
        applying the first ``d`` swap layers, the qubits form a connected
        subgraph in the current coupling map (i.e. every qubit in the tuple is
        reachable from every other within one hop).

        Results are cached both in the per-instance ``_distances`` dict and
        in the full distance tensor (if already computed for the given order).

        Args:
            nodes: A tuple of qubit indices (sorted internally for caching).
                Duplicate entries return ``-1``.
            cutoff: Maximum swap depth to check.  Defaults to
                ``len(swap_layers) + 1``.

        Returns:
            The minimum number of swap layers (0-indexed) after which the
            qubits form a connected subgraph, or ``-1`` if no valid depth is
            found within ``cutoff``.
        """
        if cutoff is None:
            cutoff = len(self._swap_layers) + 1
        nodes = tuple(sorted(nodes))
        if np.any([nodes[i] == nodes[i+1] for i in range(len(nodes)-1)]):
            return -1
        distance = self._distances.get(nodes, None)
        if distance is not None:
            return distance
        distance_tensor = self._distance_tensors.get(len(nodes), None)
        if distance_tensor is not None:
            return distance_tensor[nodes]
        
        if len(nodes) < 2:
            return 0
        
        for i in range(cutoff):
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
        return -1
    
    
    def all_connected_subgraphs(self, layer: int, order: int):
        cmap = self.swapped_coupling_map(layer)
        g = [set(cmap.neighbors(q)) for q in cmap.physical_qubits]
        def _recurse(t: tuple, possible: set[int], excluded: set[int]):
            if len(t) == order:
                yield tuple(sorted(list(t)))
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
        """Return the full distance tensor for interactions of a given order.

        An ``order``-th-order distance tensor is an ``n^order`` numpy array
        (where ``n`` is the number of qubits) such that element ``[i, j, ...]``
        gives the minimum swap depth required to make qubits ``i, j, ...``
        mutually adjacent.  Order-2 tensors coincide with the standard distance
        matrix.

        Computation is lazy: the tensor is built by enumerating all connected
        subgraphs of increasing swap depth.  The result is cached in memory and
        on disk (under ``/lustre/scratch127/qpg/jc59/hubo_swap_strategies/``).
        For all-to-all topologies the tensor is filled with zeros immediately.

        Args:
            order: The interaction order (number of qubits per interaction).
                Must be >= 2.

        Returns:
            An ``n^order`` numpy int array of swap distances, with ``-1``
            for qubit tuples that cannot be made adjacent within the available
            swap layers.
        """
        if order == 2:
            return self.distance_matrix
        
        dt = self._distance_tensors.get(order, None)
        if dt is not None:
            return dt
        
        try:
            with open(f'/lustre/scratch127/qpg/jc59/hubo_swap_strategies/swap_strategy_{self._type}_distance_qubits_{self._num_vertices}_order_{order}.pkl', 'rb') as f:
                dt = pickle.load(f)
                logger.info("Loaded data")
                return dt
        except FileNotFoundError:
            logger.info('Computing data')
            pass
        except Exception as e:
            raise Exception(f'Other than file not found: {e}')
        
        
        if self._type == 'all_to_all':
            distance_tensor = np.full([self._num_vertices]*order, -1)
            np.put(
                distance_tensor, 
                [np.ravel_multi_index(x, distance_tensor.shape) for x in permutations(range(self._num_vertices), order)], 
                0
            )
            return distance_tensor
        
        
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
        
        with open(f'/lustre/scratch127/qpg/jc59/hubo_swap_strategies/swap_strategy_{self._type}_distance_qubits_{self._num_vertices}_order_{order}.pkl', 'wb') as f:
            pickle.dump(distance_tensor, f)
        
        return distance_tensor