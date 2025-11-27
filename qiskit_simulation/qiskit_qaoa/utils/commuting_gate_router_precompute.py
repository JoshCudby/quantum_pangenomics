import numpy as np
from typing import overload, Dict, Set, Iterable, List, Generator, Optional, Tuple
from collections import deque, defaultdict
from functools import reduce
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
                physical_indices = tuple(sorted([current_layout.get_virtual_bits()[dag.qubits[i]] for i in interaction]))
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
            seq = bfs_shortest_sequence(list(all_vertices_in_chain), edges, currently_stored_info, max_states=100000)
            if seq is not None:
                for gate in seq:
                    circuit.cx(gate[0], gate[1])
                return
        
            # print('Could not find shortest sequence by BFS, try heuristic')
            seq = heuristic_spanning_tree_solver(list(all_vertices_in_chain), edges, currently_stored_info, max_local_bfs_states=2000)
            if seq is not None:
                for gate in seq:
                    circuit.cx(gate[0], gate[1])
                return
            # print('Could not find shortest sequence by heuristic')
        except Exception:
            pass   
        
        for gate in cx_gates[::-1]:
            circuit.cx(gate[0], gate[1])

    
    
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
        
        cx_gates = []
        possible_interactions = list(current_layer.keys())
        
        extra_interactions: list[tuple[int,...]] = sorted(list(impossible_gates.keys()), key=lambda e: sum(circuit.num_qubits**i * e[-i] for i in range(len(e))))
        used_extra_interactions = []
        
        
        while len(current_layer):
            currently_stored_info = {x: {x} for x in range(circuit.num_qubits)}
            all_vertices_in_chain = set()
            have_chain = True
            final_interaction = None
            index = 0
            
            while have_chain:
                have_chain, applied_cx_gates, final_interaction, applied_extra_interactions = self._precompute_chain(
                    index % 2 == 1, 
                    possible_interactions, 
                    current_layer, 
                    currently_stored_info, 
                    final_interaction, 
                    extra_interactions,
                    impossible_gates,
                    circuit
                )
                cx_gates.extend(applied_cx_gates)
                used_extra_interactions.extend(applied_extra_interactions)
                all_vertices_in_chain = all_vertices_in_chain.union(final_interaction) 
                index += 1      
            
            self._reset_info(circuit, currently_stored_info, cx_gates, all_vertices_in_chain)
            cx_gates = []
            blocked_vertices = blocked_vertices.union(all_vertices_in_chain)
            possible_interactions = [interaction for interaction in possible_interactions if set(interaction).isdisjoint(blocked_vertices)]
            if len(possible_interactions) == 0:
                blocked_vertices = set()
                possible_interactions = list(current_layer.keys())                         
            
        return used_extra_interactions
    
        
    def _is_connected(
        self,
        nodes: set[int] | tuple[int,...]
    ) -> bool:
        return self._swap_strategy.distance_nodes(tuple(nodes), cutoff=1) == 0
        
        
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
        extra_interactions:  list[tuple[int,...]],
        impossible_gates: dict[tuple[int,...], Gate],
        circuit: QuantumCircuit
    ) -> tuple[bool, list[tuple[int, int]], tuple[int,...], list[tuple[int,...]]]:
        if ascending and previous_final_interaction is None:
            raise Exception('Need previous final interaction if subset chain')
        
        cx_gates = []
        final_interaction = ()
        
        
        def find_final_interaction(interactions) -> tuple[Optional[tuple[int,...]], Optional[tuple[int,...]]]:
            cx_qubits, final_interaction = None, None
            for interaction in sort_by_length(interactions, ascending=ascending):
                allowed_num_cx = len(interaction) - 1 if previous_final_interaction is None else np.abs(len(previous_final_interaction) - len(interaction))
                cx_qubits = self._is_implementable_in_linear_cx_depth(interaction, currently_stored_info, allowed_num_cx)
                if cx_qubits is not None and (not ascending or set(interaction).issubset(previous_final_interaction)): # type: ignore
                    final_interaction = interaction
                    break
            return cx_qubits, final_interaction


        # Start by finding the endpoint of the chain
        cx_qubits, final_interaction = find_final_interaction(available_interactions)
        if cx_qubits is None:
            cx_qubits, final_interaction = find_final_interaction(extra_interactions)           
            
        if cx_qubits is None or final_interaction is None:
            # No chain
            return False, cx_gates, (), []
        
        
        # Find the best set of interactions to fill in the chain
        # ie find the CX network to build final_interaction from first_interaction
        edges = [c for c in combinations(cx_qubits, 2) if self._is_connected(c)]
        possible_cx_sequences = enumerate_removal_pair_sequences(
            cx_qubits,
            edges,
            max_solutions=100
        )
        best_num_interactions = 0
        best_sequence = None
        all_interactions = available_interactions + extra_interactions
        for cx_sequence in possible_cx_sequences:
            currently_stored_info_copy = currently_stored_info.copy()
            num_interactions = 0
            for cx in cx_sequence:
                currently_stored_info_copy[cx[1]] = currently_stored_info_copy[cx[1]].symmetric_difference(currently_stored_info_copy[cx[0]])
                if tuple(sorted(list(currently_stored_info_copy[cx[1]]))) in all_interactions: 
                    num_interactions += 1
            if num_interactions > best_num_interactions:
                best_sequence = cx_sequence
                best_num_interactions = num_interactions
        
        if best_sequence is None:
            raise Exception('Failed to find best sequence of CX gates')
        
        
        def apply_interaction(qubit: int, interaction: tuple[int,...], interactions_list: list[tuple[int,...]], gate_dict: dict[tuple[int,...], Gate]):
            interactions_list.remove(interaction)
            gate = gate_dict.pop(interaction)
            coeff = 2 * np.real_if_close(gate.params)[0]
            circuit.rz(coeff, qubit)
        
        # Apply that sequence
        applied_extra_interactions = []
        for cx in best_sequence:
            circuit.cx(cx[0], cx[1])
            currently_stored_info[cx[1]] = currently_stored_info[cx[1]].symmetric_difference(currently_stored_info[cx[0]])
            
            # Need to sort these tuples!
            interaction = tuple(sorted(list(currently_stored_info[cx[1]])))
            if interaction in available_interactions: 
                apply_interaction(cx[1], interaction, available_interactions, current_layer)
            elif interaction in extra_interactions: 
                apply_interaction(cx[1], interaction, extra_interactions, impossible_gates)
                applied_extra_interactions.append(interaction)
                
        return True, best_sequence, final_interaction, applied_extra_interactions
        