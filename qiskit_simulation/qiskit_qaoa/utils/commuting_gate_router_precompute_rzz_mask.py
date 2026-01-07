import numpy as np
from typing import Optional, Tuple, List
from collections import defaultdict
from itertools import combinations

from qiskit import QuantumCircuit

from qiskit.circuit import Gate, Qubit
from qiskit.dagcircuit import DAGCircuit, DAGOpNode

from qiskit.transpiler import TransformationPass, generate_preset_pass_manager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout

from qiskit.converters import dag_to_circuit, circuit_to_dag

from qiskit_qaoa.utils.transpiler_passes import CommutingBlock
from qiskit_qaoa.utils.swap_strategy import ExtendedSwapStrategy
from qiskit_qaoa.utils.shortest_sequence_graph_reset import heuristic_spanning_tree_solver, bfs_shortest_sequence, sort_by_length, enumerate_removal_pair_sequences


class CommutingGateRouterPrecomputeRzz(TransformationPass):
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
        self._adj = [[j for j in range(self._num_qubits) if self._is_connected((i,j))] for i in range(self._num_qubits)]
        
        
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
        max_distance: int = int(max([x for x in gate_layers.keys() if x < np.inf] + [0]))
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
                physical_indices = tuple(sorted([current_layout.get_virtual_bits()[dag.qubits[i]] for i in interaction]))
                try:
                    impossible_nodes.pop(physical_indices)
                except KeyError as e:
                    print(impossible_nodes.keys())
                    print(interaction, physical_indices)
                    raise e
            
            
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
    
        
    def _is_connected(
        self,
        nodes: set[int] | tuple[int,...]
    ) -> bool:
        return self._swap_strategy.distance_nodes(tuple(nodes), cutoff=1) == 0
        
     

    def _make_op_layers(
        self, dag: DAGCircuit, op: CommutingBlock, layout: Layout, swap_strategy: ExtendedSwapStrategy
    ) -> tuple[dict[int, dict[tuple[int,...], Gate]], dict[tuple, DAGOpNode]]:
        """Creates layers of two-qubit gates based on the distance in the swap strategy."""

        gate_layers: dict[int, dict[tuple, Gate]] = defaultdict(dict)
        impossible_gates: dict[tuple, DAGOpNode] = {}

        for node in op.node_block:
            edge = tuple(sorted([dag.find_bit(node.qargs[i]).index for i in range(len(node.qargs))]))

            v_bits = layout.get_virtual_bits()        
            bits = tuple(sorted([v_bits[dag.qubits[edge[i]]] for i in range(len(edge))]))

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

        try:
            if circuit.num_qubits > 20:
                raise Exception('Too large - skip')
            seq = bfs_shortest_sequence(list(all_vertices_in_chain), edges, currently_stored_info, max_states=100000)
            if seq is not None and len(seq) < len(cx_gates):
                for gate in seq:
                    circuit.cx(gate[0], gate[1])
                return
            
            seq = heuristic_spanning_tree_solver(list(all_vertices_in_chain), edges, currently_stored_info, max_local_bfs_states=2000)
            if seq is not None and len(seq) < len(cx_gates):
                for gate in seq:
                    circuit.cx(gate[0], gate[1])
                return
        except Exception as e:
            # print(f'Error in bfs or heuristic: {e}')
            pass
        
        for gate in cx_gates[::-1]:
            circuit.cx(gate[0], gate[1])
    
    
    def _build_chain_sub_layers(
        self,
        current_layer: dict[tuple[int, ...], Gate],
        circuit: QuantumCircuit,
        impossible_gates: dict[tuple[int, ...], Gate]
    ) -> list:
        """
        Optimized implementation; corrected to:
        - call _reset_info with the updated currently-stored info derived from qmask_list,
        - accumulate blocked_vertices across outer iterations and only reset it when we repopulate possible_interactions.
        """

        gate = current_layer.pop(tuple(), None)
        if gate is not None:
            circuit.global_phase = circuit.global_phase - 0.5 * np.real_if_close(gate.params)[0]

        one_qubit_sites = [k for k in list(current_layer.keys()) if len(k) == 1]
        for site in one_qubit_sites:
            gate = current_layer.pop(site)
            coeff = 2 * np.real_if_close(gate.params)[0]
            q_index = site[0] if isinstance(site, tuple) else site
            circuit.rz(coeff, q_index)

        universe_elements = set()
        for k in current_layer.keys():
            universe_elements.update(k)
        for k in impossible_gates.keys():
            universe_elements.update(k)
        universe_elements.update(range(circuit.num_qubits))
        element_to_bit = {e: i for i, e in enumerate(sorted(universe_elements))}
        bit_to_element = {i: e for e, i in element_to_bit.items()}

        def iter_to_mask_local(iterable: tuple[int, ...] | set[int]) -> int:
            m = 0
            for e in iterable:
                m |= 1 << element_to_bit[e]
            return m

        def mask_to_tuple_local(mask: int) -> tuple[int, ...]:
            if mask == 0:
                return tuple()
            elems = []
            b = mask
            idx = 0
            while b:
                if b & 1:
                    elems.append(bit_to_element[idx])
                b >>= 1
                idx += 1
            return tuple(sorted(elems))

        available_mask_set = set(iter_to_mask_local(k) for k in current_layer.keys())
        mask_to_gate_current = {iter_to_mask_local(k): g for k, g in current_layer.items()}
        extra_mask_set = set(iter_to_mask_local(k) for k in impossible_gates.keys())
        mask_to_gate_impossible = {iter_to_mask_local(k): g for k, g in impossible_gates.items()}
        all_mask_set = available_mask_set | extra_mask_set
        mask_to_tuple_map = {m: mask_to_tuple_local(m) for m in all_mask_set}

        n_qubits = circuit.num_qubits

        def _is_implementable_in_linear_cx_depth_mask(
            interaction_mask: int,
            qmask_list: List[int],
            allowed_number_of_cx: int,
        ) -> Optional[Tuple[int, ...]]:
            n = len(qmask_list)
            max_k = allowed_number_of_cx + 2
            if max_k < 1:
                return None
            for q in range(n):
                if qmask_list[q] == interaction_mask and 1 <= max_k:
                    return (q,)
            def frontier_search(qubits: tuple[int,...]):
                for start in qubits:
                    cur_list = [start]
                    cur_mask = qmask_list[start]
                    frontier = [nb for nb in self._adj[start] if nb != start and nb > start] 
                    stack = [(cur_list, cur_mask, frontier)]
                    while stack:
                        s_list, s_mask, s_front = stack.pop()
                        if len(s_list) >= max_k:
                            continue
                        for nb in s_front:
                            if nb in s_list:
                                continue
                            new_list = s_list + [nb]
                            new_mask = s_mask ^ qmask_list[nb]
                            if new_mask == interaction_mask:
                                return tuple(sorted(new_list))
                            if len(new_list) < max_k:
                                new_front = []
                                for x in s_front:
                                    if x != nb and x not in new_list:
                                        new_front.append(x)
                                for nb2 in self._adj[nb]:
                                    if nb2 > start and nb2 not in new_list and nb2 not in new_front:
                                        new_front.append(nb2)
                                stack.append((new_list, new_mask, new_front))
                return None
            cx_qubits = frontier_search(mask_to_tuple_local(interaction_mask))
            if cx_qubits is not None:
                return cx_qubits
            return frontier_search(range(n))

        def apply_rzz_from_mask(q1: int, q2: int, mask: int, from_extra: bool, applied_extra_masks: list[int]):
            if from_extra:
                gate = mask_to_gate_impossible.pop(mask, None)
                tup = mask_to_tuple_map.get(mask)
                if tup is not None:
                    impossible_gates.pop(tup, None)
                extra_mask_set.discard(mask)
            else:
                gate = mask_to_gate_current.pop(mask, None)
                tup = mask_to_tuple_map.get(mask)
                if tup is not None:
                    current_layer.pop(tup, None)
                available_mask_set.discard(mask)
            if gate is None:
                return
            coeff = 2 * np.real_if_close(gate.params)[0]
            circuit.rzz(coeff, q1, q2)
            if from_extra:
                applied_extra_masks.append(mask)

        def apply_rz_from_mask(q: int, mask: int, from_extra: bool, applied_extra_masks: list[int]):
            if from_extra:
                gate = mask_to_gate_impossible.pop(mask, None)
                tup = mask_to_tuple_map.get(mask)
                if tup is not None:
                    impossible_gates.pop(tup, None)
                extra_mask_set.discard(mask)
            else:
                gate = mask_to_gate_current.pop(mask, None)
                tup = mask_to_tuple_map.get(mask)
                if tup is not None:
                    current_layer.pop(tup, None)
                available_mask_set.discard(mask)
            if gate is None:
                return
            coeff = 2 * np.real_if_close(gate.params)[0]
            circuit.rz(coeff, q)
            if from_extra:
                applied_extra_masks.append(mask)

        used_extra_interactions_masks: list[int] = []

        blocked_vertices: set[int] = set()

        safety_counter = 0
        SAFETY_LIMIT = 1_000

        while len(current_layer):
            safety_counter += 1
            if safety_counter > SAFETY_LIMIT:
                raise Exception("Exceeded safety limit while clearing current_layer")


            possible_interactions = [p for p in list(current_layer.keys()) if set(p).isdisjoint(blocked_vertices)]

            if len(possible_interactions) == 0:
                blocked_vertices = set()
                possible_interactions = list(current_layer.keys())

            qmask_list = [iter_to_mask_local({i}) for i in range(n_qubits)]

            all_vertices_in_chain = set()
            cx_gates_accum: list[tuple[int, int]] = []
            have_chain = True
            final_interaction_mask: Optional[int] = None
            index = 0

            while have_chain:
                def _precompute_chain_mask(ascending_flag: bool, previous_final_interaction: Optional[tuple[int, ...]]):
                    def find_final_interaction_from_masks(mask_set: set[int], offset: int = 0):
                        interactions_list = [mask_to_tuple_map[m] for m in mask_set if m in mask_to_tuple_map]
                        if previous_final_interaction is not None:
                            set_prev = set(previous_final_interaction)
                            interactions_list = [interaction for interaction in interactions_list if len(set_prev.intersection(interaction))]
                        for interaction_tuple in sort_by_length(interactions_list, ascending=ascending_flag):
                            # if len(interaction_tuple) == 2 and ascending_flag is True:
                            #     pass
                            allowed_num_cx = ((
                                len(interaction_tuple) - 1 if previous_final_interaction is None else abs(len(previous_final_interaction) - len(interaction_tuple))
                            )) + offset
                            interaction_mask = iter_to_mask_local(interaction_tuple)
                            cx_qubits = _is_implementable_in_linear_cx_depth_mask(interaction_mask, qmask_list, allowed_num_cx)
                            if cx_qubits is not None and (not ascending_flag or (previous_final_interaction is not None and set(interaction_tuple).issubset(previous_final_interaction))):
                                return cx_qubits, interaction_mask
                        return None, None

                    cx_qubits_local, final_mask_local = find_final_interaction_from_masks(available_mask_set, offset=0)
                    if cx_qubits_local is None:
                        cx_qubits_local, final_mask_local = find_final_interaction_from_masks(extra_mask_set, offset=1)
                    if cx_qubits_local is None or final_mask_local is None:
                        return False, [], None, []

                    if len(cx_qubits_local) == 2:
                        if final_mask_local in available_mask_set:
                            apply_rzz_from_mask(cx_qubits_local[0], cx_qubits_local[1], final_mask_local, False, used_extra_interactions_masks)
                        elif final_mask_local in extra_mask_set:
                            apply_rzz_from_mask(cx_qubits_local[0], cx_qubits_local[1], final_mask_local, True, used_extra_interactions_masks)
                        else:
                            raise Exception("Bad CX qubits in precompute")
                        return True, [], final_mask_local, [final_mask_local] if final_mask_local in used_extra_interactions_masks else []

                    edges = [c for c in combinations(cx_qubits_local, 2) if c[1] in self._adj[c[0]] or c[0] in self._adj[c[1]]]
                    possible_cx_sequences = enumerate_removal_pair_sequences(cx_qubits_local, edges, stop_at=2, max_solutions=100)

                    best_seq = None
                    best_score = -1
                    for seq in possible_cx_sequences:
                        stored_copy = qmask_list[:]
                        score = 0
                        for (a, b) in seq:
                            stored_copy[b] ^= stored_copy[a]
                            if stored_copy[b] in all_mask_set:
                                score += 1
                            for nb in self._adj[b]:
                                possible_mask = stored_copy[b] ^ stored_copy[nb]
                                if possible_mask in all_mask_set:
                                    score += 1
                        if score > best_score:
                            best_score = score
                            best_seq = seq

                    if best_seq is None:
                        return False, [], None, []

                    applied_cx_gates: list[tuple[int, int]] = []
                    applied_extra_masks_local: list[int] = []
                    for idx_seq, (a, b) in enumerate(best_seq):
                        circuit.cx(a, b)
                        applied_cx_gates.append((a, b))
                        qmask_list[b] ^= qmask_list[a]

                        cur_mask = qmask_list[b]
                        if cur_mask in mask_to_gate_current:
                            apply_rz_from_mask(b, cur_mask, False, applied_extra_masks_local)
                        elif cur_mask in mask_to_gate_impossible:
                            apply_rz_from_mask(b, cur_mask, True, applied_extra_masks_local)

                        next_pair = best_seq[idx_seq + 1] if idx_seq < len(best_seq) - 1 else (None, None)
                        for nb in self._adj[b]:
                            if nb == a or (b, nb) == next_pair:
                                continue
                            inter_mask = qmask_list[b] ^ qmask_list[nb]
                            if inter_mask in mask_to_gate_current:
                                apply_rzz_from_mask(b, nb, inter_mask, False, applied_extra_masks_local)
                            elif inter_mask in mask_to_gate_impossible:
                                apply_rzz_from_mask(b, nb, inter_mask, True, applied_extra_masks_local)

                    return True, applied_cx_gates, final_mask_local, [m for m in applied_extra_masks_local]

                have_chain, applied_cx_gates_local, final_mask_local, applied_extra_masks_local = _precompute_chain_mask(index % 2 == 1, None if final_interaction_mask is None else mask_to_tuple_local(final_interaction_mask))
                if have_chain:
                    cx_gates_accum.extend(applied_cx_gates_local)
                    used_extra_interactions_masks.extend(applied_extra_masks_local)
                    if final_mask_local is not None:
                        verts = mask_to_tuple_map.get(final_mask_local, mask_to_tuple_local(final_mask_local))
                        all_vertices_in_chain.update(verts)
                        final_interaction_mask = final_mask_local
                index += 1

            currently_stored_info_for_reset = {i: set(mask_to_tuple_local(qmask_list[i])) for i in range(n_qubits)}
            self._reset_info(circuit, currently_stored_info_for_reset, cx_gates_accum, all_vertices_in_chain)

            blocked_vertices |= set(all_vertices_in_chain)

            possible_interactions = [p for p in list(current_layer.keys()) if set(p).isdisjoint(blocked_vertices)]

            if len(possible_interactions) == 0:
                blocked_vertices = set()
                possible_interactions = list(current_layer.keys())

        if len(current_layer):
            raise Exception('Failed to clear current layer')

        used_extra_interactions = [mask_to_tuple_map[m] for m in used_extra_interactions_masks if m in mask_to_tuple_map]
        return used_extra_interactions
