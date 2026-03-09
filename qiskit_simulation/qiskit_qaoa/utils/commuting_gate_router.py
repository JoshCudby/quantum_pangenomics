from __future__ import annotations

import numpy as np
from functools import reduce
from itertools import combinations

from qiskit import QuantumCircuit

from qiskit.circuit import Gate, Qubit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.dagcircuit import DAGCircuit, DAGOpNode

from qiskit.transpiler import TransformationPass, generate_preset_pass_manager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from collections import defaultdict

from qiskit.converters import dag_to_circuit, circuit_to_dag

from qiskit_qaoa.utils.transpiler_passes import CommutingBlock
from qiskit_qaoa.utils.swap_strategy import ExtendedSwapStrategy


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
        self._max_layers: int = max_layers
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
                if len(list(accumulator.topological_op_nodes())):
                    accumulator = self._compose_non_swap_nodes(accumulator, current_layout, new_dag, swap_strategy)

                # Decompose the swap-strategy node and add to the dag.
                new_dag.compose(self.swap_decompose(dag, node, current_layout, swap_strategy))
            else:
                print('Not commuting block')
                accumulator.apply_operation_back(node.op, node.qargs, node.cargs)
        
        print(f'Gates we cannot directly implement: {len(cannot_implement)}')
        print([tuple(sorted([dag.find_bit(sub_node.qargs[i]).index for i in range(len(sub_node.qargs))])) for sub_node in cannot_implement])
        
        if self._perform_extra_swaps:
            # TODO: find the best point to implement them, rather than dumping at the end i.e. the time when minimum distance for the ops
            for sub_node in cannot_implement:
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

    def _build_sub_layers(
        self, current_layer: dict[tuple[int, ...], Gate]
    ) -> list[dict[tuple[int, ...], Gate]]:
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
        self, current_layer: dict[tuple[int, ...], Gate]
    ) -> list[dict[tuple[int, ...], Gate]]:
        """The edge coloring method of building sub-layers of commuting gates."""
        if self._edge_coloring is None:
            raise Exception('No edge coloring')
        
        sub_layers: list[dict[tuple[int, int], Gate]] = [
            {} for _ in set(self._edge_coloring.values())
        ]
        greedy_gates = {}
        for edge, gate in current_layer.items():
            if len(edge) == 2:
                color = self._edge_coloring[tuple(sorted(edge))]
                try:
                    sub_layers[color][edge] = gate
                except IndexError as e:
                    raise e
            else:
                greedy_gates[edge] = gate
                
        greed_sub_layers = self._greedy_build_sub_layers_2(greedy_gates)
        sub_layers.extend(greed_sub_layers)
        return sub_layers
    
    
    def _is_connected(
        self,
        nodes: set[int]
    ) -> bool:
        if self._swap_strategy is None:
            raise Exception('No swap strategy to determine if nodes are connected')
        return self._swap_strategy.distance_nodes(tuple(nodes)) == 0
    
    
    def _missing_info_is_connected(
        self,
        missing_info: set[int],
        rotation_site: int,
        currently_stored_info: dict[int, set[int]]
    ):
        # nodes = reduce(
        #     set.symmetric_difference,
        #     [currently_stored_info[q] for q in missing_info],
        #     set()
        # )
        # nodes.add(rotation_site)
        # if self._is_connected(nodes) and any([self._is_connected(set([missing, rotation_site])) for missing in missing_info]):
        #     return True
        # print(f'Starting missing_info_is_connected. Missing: {missing_info}, site: {rotation_site}')
        
        nodes: set[int] = reduce(
            set.symmetric_difference,
            [currently_stored_info[q] for q in missing_info],
            set()
        )
        nodes.add(rotation_site)
        
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
                if any([self._is_connected(set([missing, s])) for s in subset]):
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
            
        
        if (
            self._is_connected(nodes) 
            and all([self._is_connected(subset.union([rotation_site])) for subset in missing_connected_subsets]) 
            and all(subsets_can_fix)
        ):
            return True
        return False
    
    
    def _suitable_superset(
        self,
        next_site: tuple[int,...],
        rotation_site: int | None,
        possible_sites: list[tuple[int,...]],
        currently_stored_info: dict[int, set[int]]
    ) -> tuple[tuple[int,...] | None, int | None]:
        for site in possible_sites:
            if set(site).issuperset(set(next_site)):
                missing_info = set(site).difference(set(next_site))
                if rotation_site is not None:
                    if self._missing_info_is_connected(missing_info, rotation_site, currently_stored_info):
                        return site, rotation_site
                else:
                    for x in next_site:
                        if self._missing_info_is_connected(missing_info, x, currently_stored_info):
                            return site, x
        return None, None
    
    
    def _suitable_subset(
        self,
        next_site: tuple[int,...],
        rotation_site: int,
        possible_sites: list[tuple[int,...]],
        currently_stored_info: dict[int, set[int]]
    ) -> tuple[tuple[int,...] | None, int | None]:
        for site in possible_sites[::-1]:
            if rotation_site in site and set(site).issubset(set(next_site)):
                missing_info = set(next_site).difference(set(site))
                # missing_info = set(site).difference(set(next_site))
                if self._missing_info_is_connected(missing_info, rotation_site, currently_stored_info):
                    return site, rotation_site
        return None, None


    def _compute_site_abstract(
        self,
        site: tuple[int,...],
        rotation_site: int,
        gate: Gate,
        circuit: QuantumCircuit,
        currently_stored_info: dict[int, set[int]]
    ) -> tuple[QuantumCircuit, list[tuple[int, int]]]:
        donor_qc = QuantumCircuit(circuit.num_qubits)
        circuit._append(gate, [donor_qc.qubits[i] for i in site], [])
        return circuit, []
    
   
    def _compute_site(
        self,
        site: tuple[int,...],
        rotation_site: int,
        gate: Gate,
        circuit: QuantumCircuit,
        currently_stored_info: dict[int, set[int]]
    ) -> tuple[QuantumCircuit, list[tuple[int, int]]]:
        """
        site: the set of qubits whose info should be stored in rotation_site 
        """
        if self._swap_strategy is None:
            raise Exception('No SWAP strategy to find qubit distances')
        if rotation_site not in site:
            raise Exception(f'Rotation site: {rotation_site} not in site: {site}')
        if not isinstance(gate, PauliEvolutionGate):
            raise Exception(f'Expected PauliEvolutionGate, got {gate}')
        
        # circuit.append(gate, site)
        # return circuit, []
        
        missing_information = currently_stored_info[rotation_site].symmetric_difference(set(site))
        initial_missing_information = currently_stored_info[rotation_site].symmetric_difference(set(site))
        cx_gates = []
        iter = 0        
        
        # print(f'Computing site: {site} onto {rotation_site} with coeff: {2 * np.real_if_close(gate.params)[0]}. Missing info: {missing_information}. Stored info: {currently_stored_info}')
        while len(missing_information) > 0 and iter <= 10:
            new_cx_gates = []
            candidates = [q for q in missing_information if self._is_connected(set([q, rotation_site]))]
            if len(candidates) == 0:
                print()
                print()
                print('No candidates')
                print(f'Site: {site}, rotation_site: {rotation_site}')
                print(f'Missing info: {missing_information}')
                print(f'initial_missing_information: {initial_missing_information}')
                print(f'Stored info: {currently_stored_info}')
                print(f'Applied CX gates: {cx_gates}')
                raise Exception('No candidates to apply CX from')
            candidate = candidates[0]
            new_cx_gates.append((candidate, rotation_site))
            
            # This line can break when candidate stores info of a qubit it is not connected to (e.g. {2,3,7}), meaning that qubit is not fixed properly.
            # Try to not let such interactions happen by changing missing_info_is_connected
            # stack = [(candidate, q) for q in missing_information.symmetric_difference(currently_stored_info[candidate]) if self._is_connected(set([q, candidate]))]
            stack = [(candidate, q) for q in currently_stored_info.keys() if self._is_connected(set([q, candidate]))]
            iiter = 0
            # seen_qubits = set()
            while len(stack):                   
                previous, qubit = stack.pop()
                # qubit_neighbours = set(np.nonzero(self._swap_strategy.distance_matrix[qubit, :] == 0)[0])
                info_missing_from_path = reduce(
                    set.symmetric_difference,
                    [currently_stored_info[cx[0]] for cx in new_cx_gates],
                    missing_information
                )
                
                if qubit in info_missing_from_path:
                    new_cx_gates.append((qubit, previous))
                    # seen_qubits = seen_qubits.symmetric_difference(currently_stored_info[qubit])
                    # new_stack_entries = qubit_neighbours.intersection(info_missing_from_path.symmetric_difference(currently_stored_info[qubit])) # .difference(seen_qubits)
                    # stack.extend([(qubit, q) for q in new_stack_entries])
                    stack.extend([(qubit, q) for q in currently_stored_info.keys() if self._is_connected(set([q, qubit]))])
                    # seen_qubits = seen_qubits.union(new_stack_entries)
                # else:
                #     print(f'Skipping qubit: {qubit}')
                                   
                    
                if iiter == 10:
                    print()
                    print()
                    print(f'Site: {site}, rotation_site: {rotation_site}')
                    print(f'Missing info: {missing_information}')
                    print(f'Stored info: {currently_stored_info}')
                    print(f'Info missing from path: {info_missing_from_path}')
                    print(f'Stack: {stack}')
                    raise Exception('Could not clear stack.')
            
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
            print(site)
            print(missing_information)
            print(currently_stored_info)
            raise Exception('Failed to implement gate')
        
        
        for cx in cx_gates:
            circuit.cx(cx[0], cx[1])
            currently_stored_info[cx[1]] = currently_stored_info[cx[1]].symmetric_difference(currently_stored_info[cx[0]])
        
        coeff = 2 * np.real_if_close(gate.params)[0]
        circuit.rz(coeff, rotation_site)
        return circuit, cx_gates
    
    
    def _build_chain_sub_layers(
        self,
        current_layer: dict[tuple[int, ...], Gate],
        circuit: QuantumCircuit
    ):
        # print(current_layer.keys())
        gate = current_layer.pop(tuple(), None)
        if gate is not None:
            # print(f'Applying phase: {gate}')
            circuit.global_phase = circuit.global_phase - 0.5 * np.real_if_close(gate.params)[0]
            # print(f'Skipping phase: {gate}')
            # pass
            
        one_qubit_gate_sites = [key for key in current_layer.keys() if len(key) == 1]
        for site in one_qubit_gate_sites:
            gate = current_layer.pop(site)
            coeff = 2 * np.real_if_close(gate.params)[0]
            circuit.rz(coeff, site)
            
        current_sub_layer_index = 0
        blocked_vertices: set[int] = set()
        currently_stored_info = {x: set([x]) for x in range(circuit.num_qubits)}
        all_vertices_in_chain: set[int] = set()
        next_site: tuple[int,...] | None = None
        next_gate = None
        cx_gates = []
        rotation_site = None
        possible_sites: list[tuple[int,...]] = sorted(list(current_layer.keys()), key=lambda e: sum(circuit.num_qubits**i * e[-i] for i in range(len(e))))
        
        while len(current_layer) > 0:        
            if next_site is None:
                next_site = possible_sites.pop(0)
                all_vertices_in_chain = all_vertices_in_chain.union(next_site)
                next_gate = current_layer.pop(next_site)  
                           
                            
            if rotation_site is None:
                if next_gate is None:
                    raise Exception('Expected to have a gate.')
                chain_next_site, rotation_site = self._suitable_superset(next_site, rotation_site, possible_sites, currently_stored_info)
                if chain_next_site is not None and rotation_site is not None:
                    circuit, applied_cx_gates = self._compute_site(next_site, rotation_site, next_gate, circuit, currently_stored_info)
                    # print(f'Computed site: {next_site}, rotation: {rotation_site}, cx: {applied_cx_gates}, Stored: {currently_stored_info}')
                    cx_gates.extend(applied_cx_gates)
                    
                    possible_sites.remove(chain_next_site)
                    next_site = chain_next_site
                    all_vertices_in_chain = all_vertices_in_chain.union(next_site)
                    next_gate = current_layer.pop(next_site)
                    
                else:
                    circuit, applied_cx_gates = self._compute_site(next_site, next_site[0], next_gate, circuit, currently_stored_info)
                    blocked_vertices = blocked_vertices.union(all_vertices_in_chain)
                    for cx in applied_cx_gates[::-1]:
                        circuit.cx(cx[0], cx[1])
                    rotation_site = None
                    next_site = None
                    # circuit.barrier(label=str(all_vertices_in_chain))
                    all_vertices_in_chain = set()
                    currently_stored_info = {x: set([x]) for x in range(circuit.num_qubits)}
                    
            else:
                if next_gate is None:
                    raise Exception('Expected to have a gate.')
                
                circuit, applied_cx_gates = self._compute_site(next_site, rotation_site, next_gate, circuit, currently_stored_info)
                # print(f'Computed site: {next_site}, rotation: {rotation_site}, cx: {applied_cx_gates}, Stored: {currently_stored_info}')
                # blocked_vertices = blocked_vertices.union(set(next_site))
                cx_gates.extend(applied_cx_gates)
                
                
                chain_next_site, _ = self._suitable_superset(next_site, rotation_site, possible_sites, currently_stored_info)
                if chain_next_site is not None:
                    possible_sites.remove(chain_next_site)
                    next_site = chain_next_site
                    all_vertices_in_chain = all_vertices_in_chain.union(next_site)
                    next_gate = current_layer.pop(next_site)
                    continue
                
                chain_next_site, _ = self._suitable_subset(next_site, rotation_site, possible_sites, currently_stored_info)
                if chain_next_site is not None:
                    possible_sites.remove(chain_next_site)
                    next_site = chain_next_site
                    next_gate = current_layer.pop(next_site)
                    continue
                
                blocked_vertices = blocked_vertices.union(all_vertices_in_chain)
                for cx in cx_gates[::-1]:
                    circuit.cx(cx[0], cx[1])
                cx_gates = []
                currently_stored_info = {x: set([x]) for x in range(circuit.num_qubits)}
                rotation_site = None
                next_site = None
                next_gate = None
                # circuit.barrier(label=str(all_vertices_in_chain))
                all_vertices_in_chain = set()
            
            possible_sites = [site for site in possible_sites if set(site).isdisjoint(blocked_vertices)]
            if len(possible_sites) == 0:
                blocked_vertices = set()
                possible_sites: list[tuple[int,...]] = sorted(list(current_layer.keys()), key=lambda e: sum(circuit.num_qubits**i * e[-i] for i in range(len(e))))
                
                current_sub_layer_index += 1
        
        if next_site is not None and rotation_site is not None and next_gate is not None:
            circuit, applied_cx_gates = self._compute_site(next_site, rotation_site, next_gate, circuit,currently_stored_info)
            # print(f'Computed site: {next_site}, rotation: {rotation_site}, cx: {applied_cx_gates}, Stored: {currently_stored_info}')
            cx_gates.extend(applied_cx_gates)
             
            for cx in cx_gates[::-1]:
                circuit.cx(cx[0], cx[1])
            # circuit.barrier(label=str(all_vertices_in_chain))
        
        # print(f'Number of sub layers: {current_sub_layer_index}')
        return
    
    
    @staticmethod
    def _greedy_build_sub_layers_2(
        current_layer: dict[tuple[int,...], Gate]
    ) -> list[dict[tuple[int,...], Gate]]:
        """The greedy method of building sub-layers of commuting gates.
        We have multi-qubit Z rotations, which need to be decomposed into multiple layers, throughout all of which each qubit is blocked out.
        
        """
        sub_layers: list[dict[tuple[int,...], Gate]] = []
        current_sub_layer_index = 0
        while len(current_layer) > 0:
            current_sub_layer: dict[tuple[int,...], Gate] = {}
            remaining_gates: dict[tuple[int,...], Gate] = {}
            blocked_vertices: dict[int, set[tuple]] = {}

            for edge, evo_gate in current_layer.items():
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
        print(f'Max layers needed to apply swap decompose: {max_distance}')

        circuit_with_swap = QuantumCircuit(len(dag.qubits))

        for i in range(max_distance + 1):
            # Get current layer and replace the problem indices j,k by the corresponding
            # positions in the coupling map. The current layer corresponds
            # to all the gates that can be applied at the ith swap layer.
            current_layer = {}
            for indices, local_gate in gate_layers.get(i, {}).items():
                current_layer[self._position_in_cmap(dag, indices, current_layout)] = local_gate


            # OLD: #######################################################

            # Not all gates that are applied at the ith swap layer can be applied at the same
            # time. We therefore greedily build sub-layers.
            # print(f'Layer {i}. Gates in layer: {current_layer.keys()}')
            # print(f'Unmapped gates in layer: {[indices for indices, _ in gate_layers.get(i, {}).items()]}')
            # sub_layers = self._build_sub_layers(current_layer)
            # print(f'Layer {i}. Sub-layers: {len(sub_layers)}. Max interaction size: {max([len(key) for key in current_layer.keys()]) if len(current_layer.keys()) > 0 else 0}')

            # Apply sub-layers
            # for sublayer in sub_layers:
            #     for edge, local_gate in sublayer.items():
            #         circuit_with_swap.append(local_gate, edge)
            #######################################################
                    
                    
            self._build_chain_sub_layers(
                current_layer,
                circuit_with_swap
            )
            
            
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
            
