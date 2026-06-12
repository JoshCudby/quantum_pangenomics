"""Updated routing implementation using greedy parity networks.

Provides ``CommutingGateRouterNew``, an updated alternative to
``CommutingGateRouter`` that replaces the chain-based CX+RZ decomposition with
the explicit parity-network approach from ``routing_utils``:

- ``greedy_parity_network`` computes the CX-and-RZ sequence that implements
  all gates in the current layer, building a GF(2) parity matrix.
- ``greedy_gaussian_elimination`` then uncomputes the parity matrix back to
  the identity.

This differs from ``commuting_gate_router.py`` in that it does not maintain a
running ``currently_stored_info`` dict between gates; instead it resets parity
tracking to the identity at the start of every layer.  The approach is more
transparent but may produce longer CX sequences for large interaction sets.
"""

import numpy as np
from collections import defaultdict

from qiskit import QuantumCircuit

from qiskit.circuit import Gate, Qubit
from qiskit.dagcircuit import DAGCircuit, DAGOpNode

from qiskit.transpiler import TransformationPass, generate_preset_pass_manager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout

from qiskit.converters import dag_to_circuit, circuit_to_dag

from qiskit_qaoa.utils.transpiler_passes import CommutingBlock
from qiskit_qaoa.utils.swap_strategy import ExtendedSwapStrategy
from qiskit_qaoa.utils.routing_utils import greedy_parity_network, greedy_gaussian_elimination


class CommutingGateRouterNew(TransformationPass):
    """Updated router using explicit greedy parity networks and Gaussian elimination.

    Routes commuting Pauli-Z evolution gates by encoding the current layer as a
    parity-vector dict and calling ``greedy_parity_network`` to decompose it
    into CX + RZ gates, followed by ``greedy_gaussian_elimination`` to uncompute
    the parity transformation.  Unlike the chain-based routers this approach
    resets parity state fully between layers.
    """

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
        """Apply the greedy-parity-network routing pass to the DAG.

        Iterates topologically over ``dag``.  Each ``CommutingBlock`` is
        replaced by its routed CX/RZ decomposition (via ``swap_decompose``).
        Non-block nodes accumulate in a side DAG; if ``perform_extra_swaps`` is
        True those are compiled with a Qiskit preset pass manager at
        optimisation level 3 and appended to the output.

        Args:
            dag: Input ``DAGCircuit`` with a single quantum register.

        Returns:
            A new ``DAGCircuit`` with ``CommutingBlock`` nodes replaced by
            native CX/RZ gate sequences.

        Raises:
            TranspilerError: If the circuit has more than one quantum register
                or contains qubits outside the register.
        """
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
            basis_gates=['rz', 'cx', 'cz', 'id', 'swap','u'],
            initial_layout=layout,
        )
        pm.layout = None
        print('Transpiling un-implemented gates')

        compiled_circuit_dag = circuit_to_dag(pm.run(dag_to_circuit(temp_dag)))

        new_dag.compose(compiled_circuit_dag, qubits=new_dag.qubits)
        
        
    def swap_decompose(
        self, dag: DAGCircuit, node: DAGOpNode, current_layout: Layout, swap_strategy: ExtendedSwapStrategy
    ) -> DAGCircuit:
        """Decompose a single ``CommutingBlock`` node using greedy parity networks.

        Builds swap-layer-indexed gate layers from the block, then for each
        layer calls ``_build_sub_layers`` (which invokes ``greedy_parity_network``
        and ``greedy_gaussian_elimination``) to emit CX/RZ gates, followed by
        SWAP gates to advance the layout.

        Args:
            dag: The parent ``DAGCircuit`` (used for qubit look-ups).
            node: The ``DAGOpNode`` whose ``op`` is a ``CommutingBlock``.
            current_layout: Virtual-to-physical qubit mapping; updated in-place
                by each SWAP gate applied.
            swap_strategy: The ``ExtendedSwapStrategy`` supplying swap-layer
                sequences and distance information.

        Returns:
            A ``DAGCircuit`` implementing the block under the current layout via
            CX/RZ and SWAP gates.
        """
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

            applied_impossible_interactions = self._build_sub_layers(
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
    
    
    def _make_op_layers(
        self, dag: DAGCircuit, op: CommutingBlock, layout: Layout, swap_strategy: ExtendedSwapStrategy
    ) -> tuple[dict[int, dict[tuple[int,...], Gate]], dict[tuple[int,...], DAGOpNode]]:
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
       
    
    def _build_sub_layers(
        self,
        current_layer: dict[tuple[int, ...], Gate],
        circuit: QuantumCircuit,
        impossible_gates: dict[tuple[int, ...], Gate]  
    ) -> list:
        vector_current_layer = {tuple([int(z in x) for z in range(circuit.num_qubits)]): val for x, val in current_layer.items()}
        A = greedy_parity_network(vector_current_layer, circuit)
        greedy_gaussian_elimination(A, circuit)
        return []
    
        
    def _is_connected(
        self,
        nodes: set[int] | tuple[int,...]
    ) -> bool:
        return self._swap_strategy.distance_nodes(tuple(nodes), cutoff=1) == 0
        
        
    
    