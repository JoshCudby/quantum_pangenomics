import numpy as np
from typing import overload, Dict, Set, Iterable, List, Generator, Hashable, Optional, Tuple
from collections import deque, defaultdict
from functools import reduce
from itertools import combinations

from qiskit import QuantumCircuit

from qiskit.circuit import Gate, Qubit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.dagcircuit import DAGCircuit, DAGOpNode

from qiskit.transpiler import TransformationPass, generate_preset_pass_manager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout

from qiskit.converters import dag_to_circuit, circuit_to_dag

from qiskit_qaoa.utils.transpiler_passes import CommutingBlock
from qiskit_qaoa.utils.swap_strategy import ExtendedSwapStrategy
"""
Idea is to find whole chains in advance (like how factoring works)
Also need not fix the rotation site in advance, but using currently_stored_info can have fluid sites

Pick largest interaction - 1st peak in chain.
Build back to start of chain, picking largest subset each time.
(Or just build forward from smallest)

Alternate sequence of subsets, sequence of supersets as long as possible.
When no chain possible, reset, block vertices etc.

Compute reset CX network directly from stored_info rather than from history of applied CX? 
"""


"""
Problem with current implementation:
not actually pre-computing whole chain
so not thinking about where to put info
should be able to increase length of chains with some thought
"""

@overload
def sort_by_length(items: list[set[int]], max_val: int, ascending: bool = True) -> list[set[int]]:
    pass

@overload
def sort_by_length(items: list[tuple[int,...]], max_val: int, ascending: bool = True) -> list[tuple[int,...]]:
    pass

def sort_by_length(items, max_val: int, ascending: bool = True):
    return sorted(items, key=lambda e: (1 if ascending else -1) * sum(max_val**i * list(e)[-i] for i in range(len(e))))


def _bfs_connected(remaining: Set[int], adj: Dict[int, Set[int]]) -> bool:
    """Return True if the induced subgraph on 'remaining' is connected.
       Uses BFS/DFS restricted to 'remaining'."""
    if not remaining:
        return True  # define empty graph as connected for convenience
    start = next(iter(remaining))
    q = deque([start])
    seen = {start}
    while q:
        u = q.popleft()
        # only traverse neighbors inside remaining
        for w in adj[u]:
            if w in remaining and w not in seen:
                seen.add(w)
                q.append(w)
    return len(seen) == len(remaining)


def enumerate_removal_pair_sequences(
    vertices: Iterable[int],
    edges: Iterable[Iterable[int]],
    stop_at: int = 1,
    order: Optional[Iterable[int]] = None,
    neighbor_order: Optional[Dict[int, Iterable[int]]] = None,
    max_solutions: Optional[int] = None,
) -> Generator[List[Tuple[int, int]], None, None]:
    """
    Enumerate sequences of vertex-removal pairs (v, u) where v is removed and u is
    a neighbour of v that remains after removal (u in remaining \\ {v}).
    At each step the remaining induced subgraph must be connected.

    Parameters
    ----------
    vertices : iterable of vertex ids
    edges : iterable of (u, v) pairs (undirected)
    stop_at : int
        Stop when 'stop_at' vertices remain (default 1).
    order : optional iterable specifying the order to try candidate removals at each step.
    neighbor_order : optional dict mapping a vertex -> iterable of neighbors order to try for pairing.
    max_solutions : optional int, stop after this many sequences.

    Yields
    ------
    lists of (removed_vertex, neighbor) pairs, length = n - stop_at.
    """
    V = list(vertices)
    n = len(V)
    if stop_at < 0 or stop_at > n:
        raise ValueError("stop_at must be between 0 and number of vertices")

    # build adjacency dict
    adj: Dict[int, Set[int]] = {v: set() for v in V}
    for a, b in edges:
        if a not in adj or b not in adj:
            raise KeyError("edge references unknown vertex")
        adj[a].add(b)
        adj[b].add(a)

    # initial connectivity check
    if not _bfs_connected(set(V), adj):
        print('Initial graph not connected in pair removal')
        return  # yield nothing if initial graph isn't connected

    # connectivity cache
    conn_cache: Dict[frozenset, bool] = {}
    def is_connected(rem: Set[int]) -> bool:
        key = frozenset(rem)
        v = conn_cache.get(key)
        if v is None:
            v = _bfs_connected(rem, adj)
            conn_cache[key] = v
        return v

    candidate_order = list(order) if order is not None else list(V)
    neighbor_order_map: Dict[int, List[int]] = {}
    if neighbor_order is not None:
        for k, seq in neighbor_order.items():
            neighbor_order_map[k] = list(seq)

    remaining = set(V)
    current_pairs: List[Tuple[int, int]] = []
    solutions_found = 0

    def backtrack(remaining: Set[int], current_pairs: List[Tuple[int, int]]):
        nonlocal solutions_found
        if max_solutions is not None and solutions_found >= max_solutions:
            return
        if len(remaining) == stop_at:
            yield list(current_pairs)
            solutions_found += 1
            return

        # choose a vertex v from remaining such that remaining \ {v} is connected
        for v in candidate_order:
            if v not in remaining:
                continue
            rem2 = remaining - {v}
            # if rem2 is empty, treat as connected; but neighbor will be None
            if not is_connected(rem2):
                continue

            # get candidate neighbors (must be in rem2)
            neighs = neighbor_order_map.get(v, None)
            if neighs is None:
                neighs_iter = [u for u in adj[v] if u in rem2]
            else:
                neighs_iter = [u for u in neighs if u in rem2]

            # If rem2 is non-empty, there must be at least one neighbour in rem2 (since remaining is connected
            # before removal and size >= 2). For safety, if rem2 is empty make neighbor None.
            if not rem2:
                raise Exception('No nodes remaining in graph')
                # no remaining nodes after removal; pair neighbor = None
                # current_pairs.append((v, None))
                # yield from backtrack(rem2, current_pairs)
                # current_pairs.pop()
            else:
                for u in neighs_iter:
                    current_pairs.append((v, u))
                    yield from backtrack(rem2, current_pairs)
                    current_pairs.pop()
                    if max_solutions is not None and solutions_found >= max_solutions:
                        return

    yield from backtrack(remaining, current_pairs)
    
    
    
def labels_to_bitrows(vertices: List, labels: Dict) -> List[int]:
    """Map labels (sets of vertices) to bitmask rows in same vertex index order."""
    index = {v: i for i, v in enumerate(vertices)}
    rows = []
    for v in vertices:
        mask = 0
        for u in labels[v]:
            if u not in index:
                raise KeyError(f"Label contains unknown vertex {u}")
            mask |= (1 << index[u])
        rows.append(mask)
    return rows


def gf2_rank(rows: List[int], n: int) -> int:
    """Compute rank over GF(2) of matrix with given row bitmasks (rows length n)."""
    # Gaussian elimination on rows (in-place copy)
    A = rows[:]  # copy
    rank = 0
    for col in range(n-1, -1, -1):  # from high bit to low bit
        pivot = None
        for r in range(rank, len(A)):
            if (A[r] >> col) & 1:
                pivot = r
                break
        if pivot is None:
            continue
        # swap pivot row into position 'rank'
        A[rank], A[pivot] = A[pivot], A[rank]
        # eliminate bit from all other rows
        for r in range(len(A)):
            if r != rank and ((A[r] >> col) & 1):
                A[r] ^= A[rank]
        rank += 1
        if rank == n:
            break
    return rank

def rows_tuple(rows: List[int]) -> Tuple[int, ...]:
    return tuple(rows)

def target_identity_rows(n: int) -> Tuple[int, ...]:
    """Return tuple of rows representing identity (row i = e_i)."""
    return tuple(1 << i for i in range(n))

def bfs_shortest_sequence(
    vertices: List,
    edges: Iterable[Tuple],
    labels: Dict,
    max_states: Optional[int] = 5_000_000,
) -> Optional[List[Tuple]]:
    """
    Find a shortest sequence of operations (v, u) meaning L[v] ^= L[u],
    using BFS from the initial label matrix to the identity rows (target).
    Returns list of (v, u) in terms of vertex objects from 'vertices' list, or
    None if impossible / not found (e.g., rank < n or BFS exhausted).
    """
    index = {v: i for i, v in enumerate(vertices)}
    n = len(vertices)
    adj = {i: set() for i in range(n)}
    for a, b in edges:
        if a not in index or b not in index:
            raise KeyError("edge references unknown vertex")
        ia, ib = index[a], index[b]
        # operations allowed: row_ia ^= row_ib and row_ib ^= row_ia
        adj[ia].add(ib)
        adj[ib].add(ia)

    # convert labels to rows (bitmasks)
    rows = labels_to_bitrows(vertices, labels)  # list length n
    # quick rank check
    if gf2_rank(rows, n) < n:
        return None  # impossible

    start = rows_tuple(rows)
    goal = target_identity_rows(n)

    if start == goal:
        return []  # already solved

    # BFS
    q = deque([start])
    parent: Dict[Tuple[int, ...], Tuple[Optional[Tuple[int, ...]], Tuple[int, int]]] = {}
    parent[start] = (None, (-1, -1))  # (prev_state, operation (i,j))

    visited = 1
    while q:
        state = q.popleft()
        # reconstruct list of row ints
        cur_rows = list(state)
        # generate neighbors: for every edge (i,j) do row_i ^= row_j
        for i in range(n):
            row_i = cur_rows[i]
            if row_i == 0 and all(((cur_rows[j] >> i) & 1) == 0 for j in adj[i]):
                # optional micro-prune: if row_i is 0 and none of its neighbors has bit i,
                # then toggling row_i may be less useful — keep code simple and don't prune aggressively.
                pass
            for j in adj[i]:
                # apply operation row_i ^= row_j
                new_rows = cur_rows[:]  # copy
                new_rows[i] = new_rows[i] ^ new_rows[j]
                new_state = rows_tuple(new_rows)
                if new_state in parent:
                    continue
                parent[new_state] = (state, (i, j))
                if new_state == goal:
                    # reconstruct path
                    ops: List[Tuple[int, int]] = []
                    cur = new_state
                    while parent[cur][0] is not None:
                        prev, op = parent[cur]
                        ops.append(op)
                        cur = prev
                    ops.reverse()
                    # translate to original vertex labels
                    result = [(vertices[j], vertices[i]) for i, j in ops]
                    return result
                q.append(new_state)
                visited += 1
                if max_states is not None and visited > max_states:
                    # exhausted resource budget
                    return None
    return None



class CommutingGateRouterPrecompute(TransformationPass):
    def __init__(
        self,
        swap_strategy: ExtendedSwapStrategy,
        max_layers: int,
        perform_extra_swaps: bool = True
    ) -> None:
        super().__init__()
        self._swap_strategy = swap_strategy
        self._num_qubits = swap_strategy._num_vertices
        self._bit_indices: dict[Qubit, int] | None = None
        self._max_layers: int = max_layers
        self._perform_extra_swaps = perform_extra_swaps
        self._cannot_implement = []
        
        
    def run(self, dag: DAGCircuit) -> DAGCircuit:
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
        
        for node in dag.topological_op_nodes():
            if isinstance(node.op, CommutingBlock):
                # Decompose the swap-strategy node and add to the dag.
                new_dag.compose(self.swap_decompose(dag, node, current_layout, self._swap_strategy))
            else:
                print('Not commuting block')
                accumulator.apply_operation_back(node.op, node.qargs, node.cargs)
        
        print(f'Gates we cannot directly implement: {len(self._cannot_implement)}')
        print([tuple(sorted([dag.find_bit(sub_node.qargs[i]).index for i in range(len(sub_node.qargs))])) for sub_node in self._cannot_implement])
        
        if self._perform_extra_swaps:
            for sub_node in self._cannot_implement:
                accumulator.apply_operation_back(sub_node.op, sub_node.qargs, sub_node.cargs)
            self._compose_non_swap_nodes(accumulator, current_layout, new_dag, self._swap_strategy)
        else:
            print('Not implementing those gates')

        self.property_set["virtual_permutation_layout"] = current_layout

        return new_dag
    
    
    def _compose_non_swap_nodes(
        self, accumulator: DAGCircuit, layout: Layout, new_dag: DAGCircuit, swap_strategy: ExtendedSwapStrategy
    ) -> DAGCircuit:
        # Add all the non-swap strategy nodes that we have accumulated up to now.
        order = layout.reorder_bits(new_dag.qubits)
        order_bits: list[int | None] = [None] * len(layout)
        for idx, val in enumerate(order):
            order_bits[val] = idx

        temp_dag = new_dag.copy_empty_like()
        temp_dag.compose(accumulator, qubits=order_bits)

        
        cm = swap_strategy._coupling_map
        pm = generate_preset_pass_manager(
            optimization_level=3, 
            coupling_map=cm, 
            basis_gates=['rz', 'cx', 'cz', 'id', 'swap', 'u'],
            initial_layout=layout
        )
        init = pm.init
        pm.init = init
        pm.layout = None
        print('Transpiling un-implemented gates')

        compiled_circuit_dag = circuit_to_dag(pm.run(dag_to_circuit(temp_dag)))

        new_dag.compose(compiled_circuit_dag, qubits=new_dag.qubits)
        
        
    def swap_decompose(
        self, dag: DAGCircuit, node: DAGOpNode, current_layout: Layout, swap_strategy: ExtendedSwapStrategy
    ) -> DAGCircuit:
        trivial_layout = Layout.generate_trivial_layout(*dag.qregs.values())
        gate_layers, impossible_nodes = self._make_op_layers(dag, node.op, current_layout, swap_strategy)

        # Iterate over and apply gate layers
        max_distance: int = int(max([x for x in gate_layers.keys() if x < np.inf]))
        print(f'Max layers needed to apply swap decompose: {max_distance}')

        circuit_with_swap = QuantumCircuit(len(dag.qubits))

        for i in range(max_distance + 1):
            # Get current layer and replace the problem indices j,k by the corresponding
            # positions in the coupling map. The current layer corresponds
            # to all the gates that can be applied at the ith swap layer.
            current_layer = {}
            for indices, local_gate in gate_layers.get(i, {}).items():
                current_layer[self._position_in_cmap(dag, indices, current_layout)] = local_gate
                
            impossible_gates = {}
            for indices, node in impossible_nodes.items():
                impossible_gates[self._position_in_cmap(dag, indices, current_layout)] = node.op

            applied_impossible_interactions = self._build_chain_sub_layers(
                current_layer,
                circuit_with_swap,
                impossible_gates
            )
            for interaction in applied_impossible_interactions:
                physical_indices = tuple(current_layout.get_virtual_bits()[dag.qubits[i]] for i in interaction)
                impossible_nodes.pop(physical_indices)
            
            
            # Apply SWAP gates
            if i < max_distance:
                for swap in swap_strategy.swap_layer(i):
                    try:
                        (j, k) = [trivial_layout.get_physical_bits()[vertex] for vertex in swap]
                    except KeyError:
                        print(swap)
                        print(trivial_layout.get_physical_bits())
                        raise KeyError()

                    circuit_with_swap.swap(j, k)
                    current_layout.swap(j, k)

        self.property_set["final_layout"] = current_layout
                
        self._cannot_implement = list(impossible_nodes.values())
        return circuit_to_dag(circuit_with_swap)
    
    
    def _position_in_cmap(self, dag: DAGCircuit, indices: tuple[int,...], layout: Layout) -> tuple[int, ...]:
        return tuple(sorted([dag.find_bit(layout.get_physical_bits()[i]).index for i in indices]))
    
    
    def _make_op_layers(
        self, dag: DAGCircuit, op: CommutingBlock, layout: Layout, swap_strategy: ExtendedSwapStrategy
    ) -> tuple[dict[int, dict[tuple[int,...], Gate]], dict[tuple, DAGOpNode]]:
        """Creates layers of two-qubit gates based on the distance in the swap strategy."""

        gate_layers: dict[int, dict[tuple, Gate]] = defaultdict(dict)
        impossible_gates: dict[tuple, DAGOpNode] = {}

        for node in op.node_block:
            edge = tuple([dag.find_bit(node.qargs[i]).index for i in range(len(node.qargs))])

            v_bits = layout.get_virtual_bits()        
            bits = tuple([v_bits[dag.qubits[edge[i]]] for i in range(len(edge))])

            distance = swap_strategy.distance_nodes(bits)
            if -1 < distance <= self._max_layers:
                gate_layers[distance][edge] = node.op
            else:
                impossible_gates[edge] = node

        return gate_layers, impossible_gates
    
    
    def _reset_info(
        self,
        circuit: QuantumCircuit,
        currently_stored_info: dict[int, set[int]],
        cx_gates: list[tuple[int, int]],
        all_vertices_in_chain: set[int]
    ):
        edges = [c for c in combinations(all_vertices_in_chain, 2) if self._is_connected(c)]
        seq = bfs_shortest_sequence(list(all_vertices_in_chain), edges, currently_stored_info, max_states=500000)
        if seq is not None:
            for gate in seq:
                circuit.cx(gate[0], gate[1])
        else:
            for gate in cx_gates[::-1]:
                circuit.cx(gate[0], gate[1])
        return
    
    
    def _build_chain_sub_layers(
        self,
        current_layer: dict[tuple[int, ...], Gate],
        circuit: QuantumCircuit,
        impossible_gates: dict[tuple[int, ...], Gate]  
    ) -> list:
        gate = current_layer.pop(tuple(), None)
        if gate is not None:
            circuit.global_phase = circuit.global_phase - 0.5 * np.real_if_close(gate.params)[0]
            
        one_qubit_gate_sites = [key for key in current_layer.keys() if len(key) == 1]
        for site in one_qubit_gate_sites:
            gate = current_layer.pop(site)
            coeff = 2 * np.real_if_close(gate.params)[0]
            circuit.rz(coeff, site)
            
        blocked_vertices: set[int] = set()
        
        # previous_interaction: tuple[int,...] | None = None
        cx_gates = []
        possible_interactions = list(current_layer.keys())
        
        # TODO: think about whether extra_interaction can EVER be applied in a linear number of gates
        extra_interactions: list[tuple[int,...]] = sorted(list(impossible_gates.keys()), key=lambda e: sum(circuit.num_qubits**i * e[-i] for i in range(len(e))))
        applied_extra_interactions = []
        
        
        while len(current_layer):
            currently_stored_info = {x: {x} for x in range(circuit.num_qubits)}
            all_vertices_in_chain = set()
            have_chain = True
            final_interaction = None
            index = 0
            
            while have_chain:
                have_chain, applied_cx_gates, final_interaction = self._precompute_chain(index % 2 == 1, possible_interactions, current_layer, currently_stored_info, final_interaction, circuit)
                print(f'final_interaction: {final_interaction}, applied_cx: {applied_cx_gates}')
                cx_gates.extend(applied_cx_gates)
                all_vertices_in_chain = all_vertices_in_chain.union(final_interaction) 
                index += 1      
            
            self._reset_info(circuit, currently_stored_info, cx_gates, all_vertices_in_chain)
            cx_gates = []
            blocked_vertices = blocked_vertices.union(all_vertices_in_chain)
            possible_interactions = [interaction for interaction in possible_interactions if set(interaction).isdisjoint(blocked_vertices)]
            if len(possible_interactions) == 0:
                blocked_vertices = set()
                possible_interactions = list(current_layer.keys())                         
            
        return []
    
        
    def _is_connected(
        self,
        nodes: set[int] | tuple[int,...]
    ) -> bool:
        return self._swap_strategy.distance_nodes(tuple(nodes)) == 0
        
        
    def _is_implementable_in_linear_cx_depth(
        self,
        interaction: set[int] | tuple[int,...],
        currently_stored_info: dict[int, set[int]],
        allowed_number_of_cx: int
    ) -> tuple[int,...] | None:
        i = 1
        while i < allowed_number_of_cx + 2:
            for cx_qubits in combinations(range(self._num_qubits), i):
                if reduce(
                    set.symmetric_difference,
                    [currently_stored_info[q] for q in cx_qubits],
                    set()
                ) == set(interaction) and self._is_connected(cx_qubits):
                    return cx_qubits
            i += 1
        return None
        
    
    def _precompute_chain(
        self,
        ascending: bool,
        available_interactions: list[tuple[int,...]],
        current_layer: dict[tuple[int,...], Gate],
        currently_stored_info: dict[int, set[int]],
        previous_final_interaction: tuple[int, ...] | None,
        circuit: QuantumCircuit
    ) -> tuple[bool, list[tuple[int, int]], tuple[int,...]]:
        if ascending and previous_final_interaction is None:
            raise Exception('Need previous final interaction if subset chain')
        
        cx_gates = []
        cx_qubits = None
        final_interaction = ()

        # Start by finding the endpoint of the chain
        for interaction in sort_by_length(available_interactions, circuit.num_qubits, ascending=ascending):
            allowed_num_cx = len(interaction) - 1 if previous_final_interaction is None else np.abs(len(previous_final_interaction) - len(interaction))
            cx_qubits = self._is_implementable_in_linear_cx_depth(interaction, currently_stored_info, allowed_num_cx)
            if cx_qubits is not None and (not ascending or set(interaction).issubset(previous_final_interaction)): # type: ignore
                final_interaction = interaction
                break
        
        if cx_qubits is None:
            return False, cx_gates, ()
        
        # Find the best set of interactions to fill in the chain
        # ie find the CX network to build final_interaction from first_interaction
        edges = [c for c in combinations(cx_qubits, 2) if self._is_connected(c)]
        possible_cx_sequences = enumerate_removal_pair_sequences(
            cx_qubits,
            edges
        )
        best_num_interactions = 0
        best_sequence = None
        for cx_sequence in possible_cx_sequences:
            currently_stored_info_copy = currently_stored_info.copy()
            num_interactions = 0
            for cx in cx_sequence:
                currently_stored_info_copy[cx[1]] = currently_stored_info_copy[cx[1]].symmetric_difference(currently_stored_info_copy[cx[0]])
                if tuple(sorted(list(currently_stored_info_copy[cx[1]]))) in available_interactions: 
                    num_interactions += 1
            if num_interactions > best_num_interactions:
                best_sequence = cx_sequence
                best_num_interactions = num_interactions
        
        if best_sequence is None:
            raise Exception('Failed to find best sequence of CX gates')
        
        # Apply that sequence
        for cx in best_sequence:
            circuit.cx(cx[0], cx[1])
            currently_stored_info[cx[1]] = currently_stored_info[cx[1]].symmetric_difference(currently_stored_info[cx[0]])
            # Need to sort these tuples!
            if (interaction := tuple(sorted(list(currently_stored_info[cx[1]])))) in available_interactions: 
                print(f'Applied interaction: {interaction}')
                available_interactions.remove(interaction)
                gate = current_layer.pop(interaction)
                coeff = 2 * np.real_if_close(gate.params)[0]
                circuit.rz(coeff, cx[1])
                
        return True, best_sequence, final_interaction
        