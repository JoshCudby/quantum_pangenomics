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


class CommutingGateRouterAllToAll(TransformationPass):
    def __init__(
        self,
        swap_strategy: ExtendedSwapStrategy,
        edge_coloring: dict[tuple[int, int], int] | None = None,
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
        if swap_strategy._type != 'all_to_all':
            raise Exception(f'Expected all_to_all strategy, got {swap_strategy._type}')
        
        self._swap_strategy = swap_strategy
        self._bit_indices: dict[Qubit, int] | None = None
        self._edge_coloring = edge_coloring


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
        
        for node in dag.topological_op_nodes():
            if isinstance(node.op, CommutingBlock):

                # Compose any accumulated non-swap strategy gates to the dag
                if len(list(accumulator.topological_op_nodes())):
                    accumulator = self._compose_non_swap_nodes(accumulator, current_layout, new_dag, swap_strategy)

                # Decompose the swap-strategy node and add to the dag.
                new_dag.compose(self.swap_decompose(dag, node, current_layout, swap_strategy))
            else:
                print('Not commuting block')
                accumulator.apply_operation_back(node.op, node.qargs, node.cargs)
        
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
    
    
    def _missing_info_is_computable(
        self,
        missing_info: set[int],
        currently_stored_info: dict[int, set[int]]
    ):                 
        # Want there to be a single chain of CX gates that makes the correct missing info (i.e. no doubling back) per component
        stack = [s for s in missing_info]
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
            reduce(
                set.symmetric_difference,
                [currently_stored_info[x] for x in chain],
                set()
            )
            for chain in possible_cx_chains
        ]
        return any([info == missing_info for info in possible_infos])
                
    
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
                if self._missing_info_is_computable(missing_info, currently_stored_info):
                    return site, rotation_site if rotation_site is not None else next_site[0]
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
                if self._missing_info_is_computable(missing_info, currently_stored_info):
                    return site, rotation_site
        return None, None
    
   
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
        
        missing_information = currently_stored_info[rotation_site].symmetric_difference(set(site))
        initial_missing_information = currently_stored_info[rotation_site].symmetric_difference(set(site))
        cx_gates = []
        iter = 0
        
        while len(missing_information) > 0 and iter <= 10:
            new_cx_gates = []
            candidates = [q for q in missing_information]
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
            
            stack = [(candidate, q) for q in currently_stored_info.keys()]
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
                    stack.extend([(qubit, q) for q in currently_stored_info.keys()])
                                   
                    
                if iiter == 10:
                    print()
                    print()
                    print(f'Site: {site}, rotation_site: {rotation_site}')
                    print(f'Missing info: {missing_information}')
                    print(f'Stored info: {currently_stored_info}')
                    print(f'Info missing from path: {info_missing_from_path}')
                    print(f'Stack: {stack}')
                    raise Exception('Could not clear stack.')
            
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
        gate = current_layer.pop(tuple(), None)
        if gate is not None:
            circuit.global_phase = circuit.global_phase - 0.5 * np.real_if_close(gate.params)[0]

            
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
            if -1 < distance:
                gate_layers[distance][edge] = node.op
            else:
                raise Exception(f'Got bad gate in A2A make op layers: {node}')

        return gate_layers



            
