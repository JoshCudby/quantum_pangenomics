from __future__ import annotations

import numpy as np
from functools import reduce
from itertools import combinations

from qiskit import QuantumCircuit

from qiskit.circuit import Gate
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.dagcircuit import DAGCircuit, DAGOpNode

from qiskit.transpiler import TransformationPass, generate_preset_pass_manager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from collections import defaultdict

from qiskit.converters import dag_to_circuit, circuit_to_dag

from qiskit_qaoa.utils.transpiler_passes import CommutingBlock
from qiskit_qaoa.utils.swap_strategy import ExtendedSwapStrategy


def print_error_and_raise(site, rotation_site, missing_information, currently_stored_info, cx_gates, info_missing_from_path, stack, exception_message):
    print()
    print()
    print(f'Site: {site}, rotation_site: {rotation_site}')
    print(f'Missing info: {missing_information}')
    print(f'Stored info: {currently_stored_info}')
    if cx_gates is not None:
        print(f'Applied CX gates: {cx_gates}')
    if info_missing_from_path is not None:
        print(f'Info missing from path: {info_missing_from_path}')
    if stack is not None:
        print(f'Stack: {stack}')
    raise Exception(exception_message)


class CommutingGateRouterRzz(TransformationPass):
    def __init__(
        self,
        swap_strategy: ExtendedSwapStrategy | None = None,
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
            max_layers: An integer defining how many layers of the SWAP strategy to perform when routing.
            perform_extra_swaps: A bool defining whether to `manually` implement gates that cannot be routed by the SWAP strategy
        """
        super().__init__()
        self._swap_strategy = swap_strategy
        if max_layers is None and swap_strategy is not None:
            max_layers = len(swap_strategy)
        self._max_layers: int = max_layers
        self._perform_extra_swaps = perform_extra_swaps
        self._cannot_implement = {}

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
        # cannot_implement = []
        
        for node in dag.topological_op_nodes():
            if isinstance(node.op, CommutingBlock):

                # Check that the swap strategy creates enough connectivity for the node.
                # cannot_implement.extend(self._check_edges(dag, node, swap_strategy))

                # Compose any accumulated non-swap strategy gates to the dag
                if len(list(accumulator.topological_op_nodes())):
                    accumulator = self._compose_non_swap_nodes(accumulator, current_layout, new_dag, swap_strategy)

                # Decompose the swap-strategy node and add to the dag.
                new_dag.compose(self.swap_decompose(dag, node, current_layout, swap_strategy))
            else:
                print('Not commuting block')
                accumulator.apply_operation_back(node.op, node.qargs, node.cargs)
        
        print(f'Gates we cannot directly implement: {len(self._cannot_implement)}')
        print([tuple(sorted([dag.find_bit(sub_node.qargs[i]).index for i in range(len(sub_node.qargs))])) for sub_node in self._cannot_implement])
        
        if self._perform_extra_swaps:
            for sub_node in self._cannot_implement:
                accumulator.apply_operation_back(sub_node.op, sub_node.qargs, sub_node.cargs)
            self._compose_non_swap_nodes(accumulator, current_layout, new_dag, swap_strategy)
        else:
            print('Not implementing those gates')

        self.property_set["virtual_permutation_layout"] = current_layout

        return new_dag


    def _compose_non_swap_nodes(
        self, accumulator: DAGCircuit, layout: Layout, new_dag: DAGCircuit, swap_strategy: ExtendedSwapStrategy
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

        temp_dag = new_dag.copy_empty_like()
        temp_dag.compose(accumulator, qubits=order_bits)

        
        cm = swap_strategy._coupling_map
        pm = generate_preset_pass_manager(
            optimization_level=3, 
            coupling_map=cm, 
            basis_gates=['rz', 'cx', 'id', 'swap', 'u'],
            initial_layout=layout
        )
        init = pm.init
        # init.remove(3)
        pm.init = init
        pm.layout = None
        print('Transpiling accumulator')

        compiled_circuit_dag = circuit_to_dag(pm.run(dag_to_circuit(temp_dag)))

        new_dag.compose(compiled_circuit_dag, qubits=new_dag.qubits)
        # new_dag.compose(accumulator, qubits=order_bits)

        # Re-initialize the node accumulator
        return new_dag.copy_empty_like()

    def _position_in_cmap(self, dag: DAGCircuit, indices: tuple[int,...], layout: Layout) -> tuple[int, ...]:
        """A helper function to track the movement of virtual qubits through the swaps.

        Args:
            indices: The index of decision variables (i.e. virtual qubit).
            layout: The current layout that takes into account previous swap gates.

        Returns:
            The position in the coupling map of the virtual qubits as a tuple.
        """
        bits = tuple([dag.find_bit(layout.get_physical_bits()[i]).index for i in indices])

        return bits
    
    
    def _is_connected(
        self,
        nodes: set[int] | tuple[int,...]
    ) -> bool:
        if self._swap_strategy is None:
            raise Exception('No swap strategy to determine if nodes are connected')
        return self._swap_strategy.distance_nodes(tuple(nodes)) == 0
    
    
    def _missing_info_is_connected(
        self,
        missing_info: set[int],
        rotation_site: tuple[int, int],
        currently_stored_info: dict[int, set[int]]
    ) -> bool:
        nodes: set[int] = reduce(
            set.symmetric_difference,
            [currently_stored_info[q] for q in missing_info],
            set()
        )
        nodes = nodes.union(rotation_site)
        
        missing_copy = missing_info.copy()
        missing_connected_subsets: list[set[int]] = []
        subset: set[int] = set()
        while len(missing_copy):
            added = False
            if len(subset) == 0:
                missing = missing_copy.pop()
                subset.add(missing)
                continue
            for missing in missing_copy:
                if any([self._is_connected((missing, s)) for s in subset]):
                    subset.add(missing)
                    missing_copy.remove(missing)
                    added = True
                    break
            if not added:
                missing_connected_subsets.append(subset)
                subset = set()
                
        if len(subset):
            missing_connected_subsets.append(subset)
        # Want there to be a single chain of CX gates that makes the correct missing info (i.e. no doubling back) per component
        subsets_can_fix = [False] * len(missing_connected_subsets)
        for idx, subset in enumerate(missing_connected_subsets):
            stack = [s for s in subset]
            
            all_stored_info = set()
            while len(stack) > 0:
                s = stack.pop()
                all_stored_info.add(s)
                stack.extend([x for x in currently_stored_info[s] if x not in all_stored_info])
                
            possible_cx_chains = reduce(
                list.__add__,
                [list(combinations(all_stored_info, i)) for i in range(1, len(all_stored_info) + 1)],
                []
            )
            possible_infos = [    
                (
                    chain,
                    reduce(
                        set.symmetric_difference,
                        [currently_stored_info[x] for x in chain],
                        set()
                    )
                )
                 for chain in possible_cx_chains
            ]
            subsets_can_fix[idx] = any([info[1] == subset and self._is_connected(info[0]) for info in possible_infos])
        
        print(
            missing_info,
            currently_stored_info,
            self._is_connected(nodes) ,
            all([self._is_connected(subset.union(rotation_site)) for subset in missing_connected_subsets]), 
            all(subsets_can_fix) 
        )
                            
        if (
            self._is_connected(nodes) 
            and all([self._is_connected(subset.union(rotation_site)) for subset in missing_connected_subsets]) 
            and all(subsets_can_fix)
        ):
            return True
        return False
        # if self._is_connected(nodes) and any([self._is_connected(set([missing, x])) for missing in missing_info for x in rotation_site]):
        #     return True
    
    
    def _suitable_superset(
        self,
        next_interaction: tuple[int,...],
        rotation_site: tuple[int, int] | None,
        possible_interactions: list[tuple[int,...]],
        currently_stored_info: dict[int, set[int]]
    ) -> tuple[tuple[int,...] | None, tuple[int, int] | None]:
        for interaction in possible_interactions:
            if set(interaction).issuperset(set(next_interaction)):
                missing_info = set(interaction).difference(set(next_interaction))
                print(next_interaction, interaction)
                if rotation_site is not None:
                    if self._missing_info_is_connected(missing_info, rotation_site, currently_stored_info):
                        return interaction, rotation_site
                else:
                    for rotation_site in combinations(next_interaction, 2):
                        if self._is_connected(rotation_site) and self._missing_info_is_connected(missing_info, rotation_site, currently_stored_info):
                            return interaction, rotation_site
        return None, None
    
    
    def _suitable_subset(
        self,
        next_interaction: tuple[int,...],
        rotation_site: tuple[int, int],
        possible_interactions: list[tuple[int,...]],
        currently_stored_info: dict[int, set[int]]
    ) -> tuple[tuple[int,...] | None, tuple[int, int] | None]:
        print(f'Testing subsets for {next_interaction}: {possible_interactions}')
        for interaction in possible_interactions[::-1]:
            if set(rotation_site).issubset(set(interaction)) and set(interaction).issubset(set(next_interaction)):
                missing_info = set(next_interaction).difference(set(interaction))
                print(next_interaction, interaction)
                if self._missing_info_is_connected(missing_info, rotation_site, currently_stored_info):
                    return interaction, rotation_site
        return None, None
    
   
    def _compute_interaction(
        self,
        interaction: tuple[int,...],
        rotation_site: tuple[int, int],
        gate: Gate,
        circuit: QuantumCircuit,
        currently_stored_info: dict[int, set[int]]
    ) -> tuple[QuantumCircuit, list[tuple[int, int]]]:
        """
        site: the set of qubits whose info should be stored in rotation_site 
        """
        if self._swap_strategy is None:
            raise Exception('No SWAP strategy to find qubit distances')
        if not set(rotation_site).issubset(set(interaction)):
            raise Exception(f'Rotation site: {rotation_site} not in interaction: {interaction}')
        if not isinstance(gate, PauliEvolutionGate):
            raise Exception(f'Expected PauliEvolutionGate, got {gate}')
        
        missing_information = currently_stored_info[rotation_site[0]].symmetric_difference(
            currently_stored_info[rotation_site[1]]
        ).symmetric_difference(set(interaction))
        cx_gates = []
        iter = 0
        
        while len(missing_information) > 0 and iter <= 10:
            new_cx_gates = []
            candidates = [(q, x) for q in missing_information for x in rotation_site if self._is_connected(set([q, x]))]
            if len(candidates) == 0:
                print_error_and_raise(interaction, rotation_site, missing_information, currently_stored_info, cx_gates, None, None, 'No candidates to apply CX from')
            
            candidate, site_to_apply = candidates[0]
            new_cx_gates.append((candidate, site_to_apply))
            # stack = [(candidate, q) for q in missing_information.symmetric_difference(currently_stored_info[candidate]) if self._is_connected(set([q, candidate]))]
            stack = [(candidate, q) for q in currently_stored_info.keys() if self._is_connected(set([q, candidate]))]
            iiter = 0
            while len(stack):                   
                previous, qubit = stack.pop()
                
                info_missing_from_path = reduce(
                    set.symmetric_difference,
                    [currently_stored_info[cx[0]] for cx in new_cx_gates],
                    missing_information
                )
                
                if qubit in info_missing_from_path:
                    new_cx_gates.append((qubit, previous))
                    # new_stack_entries = qubit_neighbours.intersection(info_missing_from_path.symmetric_difference(currently_stored_info[qubit]))
                    # stack.extend([(qubit, q) for q in info_missing_from_path.symmetric_difference(currently_stored_info[qubit]) if self._is_connected(set([qubit, q]))])
                    stack.extend([(qubit, q) for q in currently_stored_info.keys() if self._is_connected(set([q, qubit]))])
                    
                if iiter == 100:
                    print_error_and_raise(interaction, rotation_site, missing_information, currently_stored_info, None, info_missing_from_path, stack, 'Could not clear stack.')
           
            
            # As written, assumes the logic is correct, rather than being a self-checker
            # fixed_qubits = reduce(
            #     set.union,
            #     [currently_stored_info[cx[0]] for cx in new_cx_gates],
            #     set()
            # )
            # missing_information = missing_information.difference(fixed_qubits)
            
            # Checks the logic
            fixed_qubits = reduce(
                set.symmetric_difference,
                [currently_stored_info[cx[0]] for cx in new_cx_gates],
                set()
            )
            missing_information = missing_information.symmetric_difference(fixed_qubits)  
            
            cx_gates.extend(new_cx_gates[::-1])
            iter += 1
            
            
        if len(missing_information) > 0:
            print(interaction)
            print(missing_information)
            print(currently_stored_info)
            raise Exception('Failed to implement gate')
        
        
        for cx in cx_gates:
            circuit.cx(cx[0], cx[1])
            currently_stored_info[cx[1]] = currently_stored_info[cx[1]].symmetric_difference(currently_stored_info[cx[0]])
        
        coeff = 2 * np.real_if_close(gate.params)[0]
        circuit.rzz(coeff, *rotation_site)
        # circuit.barrier(label=f'{interaction}')
        return circuit, cx_gates
    

    @staticmethod
    def _end_chain(
        blocked_vertices: set[int], 
        all_vertices_in_chain: set[int],
        cx_gates: list[tuple[int, int]],
        circuit: QuantumCircuit,
        chain_len: int | None = None
    ):
        if chain_len is not None:
            print(f'Len of chain: {chain_len}')
        # print(f'Ending chain: {all_vertices_in_chain}')
        
        blocked_vertices = blocked_vertices.union(all_vertices_in_chain)
        
        # Some of these gates can be avoided?
        # e.g. if a vertex is added then later removed
        for cx in cx_gates[::-1]:
            circuit.cx(cx[0], cx[1])
        # circuit.barrier(label=f'End of chain: {str(all_vertices_in_chain)}')
        cx_gates = []
        currently_stored_info = {x: set([x]) for x in range(circuit.num_qubits)}
        rotation_site = None
        next_interaction = None
        next_gate = None
        all_vertices_in_chain = set()
        return (blocked_vertices, all_vertices_in_chain, cx_gates, circuit, 
                currently_stored_info, rotation_site, next_interaction, next_gate)
    
    
    def _build_chain_sub_layers(
        self,
        current_layer: dict[tuple[int, ...], Gate],
        circuit: QuantumCircuit,
        impossible_gates: dict[tuple[int, ...], Gate]
    ) -> list[tuple[int, ...]]:
        gate = current_layer.pop(tuple(), None)
        if gate is not None:
            circuit.global_phase = circuit.global_phase - 0.5 * np.real_if_close(gate.params)[0]

            
        one_qubit_gate_sites = [key for key in current_layer.keys() if len(key) == 1]
        # print(f'One qubit gate sites: {one_qubit_gate_sites}')
        for site in one_qubit_gate_sites:
            gate = current_layer.pop(site)
            coeff = 2 * np.real_if_close(gate.params)[0]
            circuit.rz(coeff, site)

        two_qubit_gate_sites = [key for key in current_layer.keys() if len(key) == 2]
        # print(f'Two qubit gate sites: {two_qubit_gate_sites}')
        for site in two_qubit_gate_sites:
            gate = current_layer.pop(site)
            coeff = 2 * np.real_if_close(gate.params)[0]
            circuit.rzz(coeff, *site)
            

        blocked_vertices: set[int] = set()
        currently_stored_info = {x: set([x]) for x in range(circuit.num_qubits)}
        all_vertices_in_chain: set[int] = set()
        next_interaction: tuple[int,...] | None = None
        next_gate = None
        cx_gates = []
        rotation_site = None
        possible_interactions: list[tuple[int,...]] = sorted(list(current_layer.keys()), key=lambda e: sum(circuit.num_qubits**i * e[-i] for i in range(len(e))))
        extra_interactions: list[tuple[int,...]] = sorted(list(impossible_gates.keys()), key=lambda e: sum(circuit.num_qubits**i * e[-i] for i in range(len(e))))
        applied_extra_interactions = []
        # chain_len = 0

        while len(current_layer) > 0:        
            if next_interaction is None:
                # chain_len += 1
                next_interaction = possible_interactions.pop(0)
                # print(f'First interaction in chain: {next_interaction}, rotation: {rotation_site}')
                all_vertices_in_chain = all_vertices_in_chain.union(next_interaction)
                next_gate = current_layer.pop(next_interaction)  
                           
                            
            if rotation_site is None:
                if next_gate is None:
                    raise Exception('Expected to have a gate.')
                
                chain_next_interaction, rotation_site = self._suitable_superset(next_interaction, rotation_site, possible_interactions, currently_stored_info)
                if chain_next_interaction is not None and rotation_site is not None:
                    # chain_len += 1
                    # print(f'Next site: {chain_next_interaction}, rotation: {rotation_site}')
                    circuit, applied_cx_gates = self._compute_interaction(next_interaction, rotation_site, next_gate, circuit, currently_stored_info)
                    # print(f'Computed site: {next_interaction}, rotation: {rotation_site}, cx: {applied_cx_gates}, Stored: {currently_stored_info}')
                    cx_gates.extend(applied_cx_gates)
                    
                    possible_interactions.remove(chain_next_interaction)
                    next_interaction = chain_next_interaction
                    all_vertices_in_chain = all_vertices_in_chain.union(next_interaction)
                    next_gate = current_layer.pop(next_interaction)
                    continue
                
                chain_next_interaction, rotation_site = self._suitable_superset(next_interaction, rotation_site, extra_interactions, currently_stored_info)
                if chain_next_interaction is not None and rotation_site is not None:
                    # chain_len += 1
                    print('Got interaction from extra interactions')
                    circuit, applied_cx_gates = self._compute_interaction(next_interaction, rotation_site, next_gate, circuit, currently_stored_info)
                    # print(f'Computed site: {next_interaction}, rotation: {rotation_site}, cx: {applied_cx_gates}, Stored: {currently_stored_info}')
                    cx_gates.extend(applied_cx_gates)
                    
                    extra_interactions.remove(chain_next_interaction)
                    next_interaction = chain_next_interaction
                    all_vertices_in_chain = all_vertices_in_chain.union(next_interaction)
                    next_gate = impossible_gates.pop(next_interaction)
                    applied_extra_interactions.append(next_interaction)
                    continue
    
    
                first_site = next_interaction[0]
                neighbours = set(np.nonzero(self._swap_strategy.distance_matrix[first_site, :] == 0)[0]).intersection(set(next_interaction))
                neighbours.remove(first_site)
                second_site = neighbours.pop()
                circuit, applied_cx_gates = self._compute_interaction(next_interaction, (first_site, second_site), next_gate, circuit, currently_stored_info)
                cx_gates.extend(applied_cx_gates)
                (blocked_vertices, all_vertices_in_chain, cx_gates, 
                    circuit, currently_stored_info, rotation_site, 
                    next_interaction, next_gate) = self._end_chain(blocked_vertices, all_vertices_in_chain, cx_gates, circuit)
                # chain_len = 0
                    
            else:
                if next_gate is None:
                    raise Exception('Expected to have a gate.')
                
                circuit, applied_cx_gates = self._compute_interaction(next_interaction, rotation_site, next_gate, circuit, currently_stored_info)
                # print(f'Computed site: {next_interaction}, rotation: {rotation_site}, cx: {applied_cx_gates}, Stored: {currently_stored_info}')
                cx_gates.extend(applied_cx_gates)
                
                
                chain_next_interaction, _ = self._suitable_superset(next_interaction, rotation_site, possible_interactions, currently_stored_info)
                if chain_next_interaction is not None:
                    # chain_len += 1
                    # print(f'Next site: {chain_next_interaction}, rotation: {rotation_site}')
                    possible_interactions.remove(chain_next_interaction)
                    next_interaction = chain_next_interaction
                    all_vertices_in_chain = all_vertices_in_chain.union(next_interaction)
                    next_gate = current_layer.pop(next_interaction)
                    continue
                
                chain_next_interaction, _ = self._suitable_subset(next_interaction, rotation_site, possible_interactions, currently_stored_info)
                if chain_next_interaction is not None:
                    # chain_len += 1
                    # print(f'Next site: {chain_next_interaction}, rotation: {rotation_site}')
                    possible_interactions.remove(chain_next_interaction)
                    next_interaction = chain_next_interaction
                    next_gate = current_layer.pop(next_interaction)
                    continue
                
                
                
                chain_next_interaction, _ = self._suitable_superset(next_interaction, rotation_site, extra_interactions, currently_stored_info)
                if chain_next_interaction is not None:
                    # chain_len += 1
                    print('Got interaction from impossible gates')
                    extra_interactions.remove(chain_next_interaction)
                    next_interaction = chain_next_interaction
                    all_vertices_in_chain = all_vertices_in_chain.union(next_interaction)
                    next_gate = impossible_gates.pop(next_interaction)
                    applied_extra_interactions.append(next_interaction)
                    continue
                
                
                chain_next_interaction, _ = self._suitable_subset(next_interaction, rotation_site, extra_interactions, currently_stored_info)
                if chain_next_interaction is not None:
                    # chain_len += 1
                    print('Got interaction from impossible gates')
                    extra_interactions.remove(chain_next_interaction)
                    next_interaction = chain_next_interaction
                    next_gate = impossible_gates.pop(next_interaction)
                    applied_extra_interactions.append(next_interaction)
                    continue                
                


                (blocked_vertices, all_vertices_in_chain, cx_gates, 
                    circuit, currently_stored_info, rotation_site, 
                    next_interaction, next_gate) = self._end_chain(blocked_vertices, all_vertices_in_chain, cx_gates, circuit)
                # chain_len = 0
            
            
            
            possible_interactions = [interaction for interaction in possible_interactions if set(interaction).isdisjoint(blocked_vertices)]
            extra_interactions = [interaction for interaction in extra_interactions if set(interaction).isdisjoint(blocked_vertices)]
            if len(possible_interactions) == 0:
                blocked_vertices = set()
                possible_interactions: list[tuple[int,...]] = sorted(list(current_layer.keys()), key=lambda e: sum(circuit.num_qubits**i * e[-i] for i in range(len(e))))
                extra_interactions: list[tuple[int,...]] = sorted(list(impossible_gates.keys()), key=lambda e: sum(circuit.num_qubits**i * e[-i] for i in range(len(e))))
                
        
        if next_interaction is not None and rotation_site is not None and next_gate is not None:
            circuit, applied_cx_gates = self._compute_interaction(next_interaction, rotation_site, next_gate, circuit, currently_stored_info)
            cx_gates.extend(applied_cx_gates)
            
            chain_next_interaction = None
            print('Starting end of layer seek for impossible gates')
            while(len(extra_interactions) > 0):
                if chain_next_interaction is not None:
                    # Only happens after the first loop
                    circuit, applied_cx_gates = self._compute_interaction(next_interaction, rotation_site, next_gate, circuit, currently_stored_info)
                    cx_gates.extend(applied_cx_gates)
                    
                chain_next_interaction, _ = self._suitable_superset(next_interaction, rotation_site, extra_interactions, currently_stored_info)
                if chain_next_interaction is not None:
                    # chain_len += 1
                    print('Got interaction from impossible gates')
                    extra_interactions.remove(chain_next_interaction)
                    next_interaction = chain_next_interaction
                    all_vertices_in_chain = all_vertices_in_chain.union(next_interaction)
                    next_gate = impossible_gates.pop(next_interaction)
                    applied_extra_interactions.append(next_interaction)
                    continue
                
                
                chain_next_interaction, _ = self._suitable_subset(next_interaction, rotation_site, extra_interactions, currently_stored_info)
                if chain_next_interaction is not None:
                    # chain_len += 1
                    print('Got interaction from impossible gates')
                    extra_interactions.remove(chain_next_interaction)
                    next_interaction = chain_next_interaction
                    next_gate = impossible_gates.pop(next_interaction)
                    applied_extra_interactions.append(next_interaction)
                    continue   
                
                break
            
            
            if chain_next_interaction is not None:
                circuit, applied_cx_gates = self._compute_interaction(next_interaction, rotation_site, next_gate, circuit, currently_stored_info)
                cx_gates.extend(applied_cx_gates)                              
            
            self._end_chain(blocked_vertices, all_vertices_in_chain, cx_gates, circuit)
        
        return applied_extra_interactions


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
        max_distance = max([x for x in gate_layers.keys() if x < np.inf])
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
            for indices, local_gate in gate_layers.get(np.inf, {}).items():
                impossible_gates[self._position_in_cmap(dag, indices, current_layout)] = local_gate
       
            applied_impossible_interactions = self._build_chain_sub_layers(
                current_layer,
                circuit_with_swap,
                impossible_gates
            )
            for interaction in applied_impossible_interactions:
                physical_indices = tuple(current_layout.get_virtual_bits()[dag.qubits[i]] for i in interaction)
                gate_layers.get(np.inf, {}).pop(physical_indices)
            
            
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
        self._cannot_implement = gate_layers.get(np.inf, {}).values()
        # circuit_with_swap.barrier()
        return circuit_to_dag(circuit_with_swap)


    def _make_op_layers(
        self, dag: DAGCircuit, op: CommutingBlock, layout: Layout, swap_strategy: ExtendedSwapStrategy
    ) -> dict[int, dict[tuple[int,...], Gate]]:
        """Creates layers of two-qubit gates based on the distance in the swap strategy."""

        gate_layers: dict[int, dict[tuple, Gate]] = defaultdict(dict)

        for node in op.node_block:
            edge = tuple([dag.find_bit(node.qargs[i]).index for i in range(len(node.qargs))])

            v_bits = layout.get_virtual_bits()        
            bits = tuple([v_bits[dag.qubits[edge[i]]] for i in range(len(edge))])

            distance = swap_strategy.distance_nodes(bits)
            if -1 < distance <= self._max_layers:
                gate_layers[distance][edge] = node.op
            else:
                gate_layers[np.inf][edge] = node.op

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
            if distance < 0 or distance > self._max_layers:
                cannot_implement.append(sub_node)
                # raise TranspilerError(
                #     f"{swap_strategy} cannot implement operator on {bits}."
                # )
        return cannot_implement
            
