from __future__ import annotations

import numpy as np
from typing import Tuple
from collections.abc import Iterable


from qiskit import QuantumCircuit

from qiskit.circuit import Gate, Qubit, Clbit
from qiskit.circuit.library import PauliEvolutionGate, CXGate, RZZGate
from qiskit.dagcircuit import DAGCircuit, DAGOpNode

from qiskit.transpiler import TransformationPass, generate_preset_pass_manager
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from collections import defaultdict

from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.converters import dag_to_circuit, circuit_to_dag

from qiskit.exceptions import QiskitError

from qiskit_qaoa.utils.swap_strategy import ExtendedSwapStrategy
from qiskit_qaoa.utils.logging import get_logger

logger = get_logger(__name__)
rng = np.random.default_rng(seed=1)
old_property_set = {}


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
            QiskitError: If the nodes in the node block have classical bits.
        """
        qubits: set[Qubit] = set()
        cbits: set[Clbit] = set()
        for node in node_block:
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
            elif isinstance(node.op, PauliEvolutionGate) and len(node.qargs) == 1:
                new_dag.apply_operation_front(node.op, node.qargs, node.cargs)
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
        if sum(node.op.operator.paulis[0].z) > 2 and not any(
                    np.all(np.linalg.matrix_power(
                        np.array([[self._coupling_map.distance(qubit1, qubit2) <= 1 for qubit2 in qubits] for qubit1 in qubits]), 
                        len(qubits)
                    ), axis=0)
                ):
            print(qubits) 
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
                    print(node.label)
                    print(f'Initial qubits: {qubits}')
                    print(f'Current qubits: {qubits_copy}')
                    print(f'Qubit: {qubit}')
                    print(f'Neighbours: { [[self._coupling_map.distance(qubit1, qubit2) == 1 for qubit2 in qubits] for qubit1 in qubits]}')
                    raise Exception('Disconnected qubit in decomposition')
                if num_neighbours == 1:
                    neighbour = qubits_copy[neighbours.index(True)]
                    sub_dag.apply_operation_back(CXGate(), [dag.qubits[qubit], dag.qubits[neighbour]])
                    cx_gates.append((qubit, neighbour))
                    qubits_copy.remove(qubit)
                    applied = True
                    break
            if not applied:     
                qubit = qubits_copy[0]
                neighbours = [self._coupling_map.distance(qubit, qubit2) == 1 for qubit2 in qubits_copy]
                neighbour = qubits_copy[neighbours.index(True)]
                sub_dag.apply_operation_back(CXGate(), [dag.qubits[qubit], dag.qubits[neighbour]])
                # logger.warning(f'No single neighbour qubit found. Apply a random cx to break loops. Chosen: {qubit, neighbour}')
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
                        print(e)
                        print(node.num_qubits)
                        print(node.qargs)
                        print(node.op)
                        print(node.op.num_qubits)
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

        print(order)
        print(order_bits)

        temp_dag = new_dag.copy_empty_like()
        temp_dag.compose(accumulator, qubits=order_bits)

        
        def callback_func(**kwargs):
            global old_property_set
            pass_ = kwargs['pass_']
            dag = kwargs['dag']
            time = kwargs['time']
            new_property_set: dict = kwargs['property_set'].copy()
            difference = set(new_property_set.keys()).difference(set(old_property_set.keys()))
            modified = set([key for key in old_property_set.keys() if new_property_set.get(key, None) != old_property_set.get(key, None)])
            to_print = {key: new_property_set.get(key, None) for key in difference.union(modified)}
            
            old_property_set = new_property_set

            count = kwargs['count']
            print(pass_.name(), dag.depth(), to_print, count)

        cm = swap_strategy._coupling_map
        layout_cm = CouplingMap(
            [(order[edge[0]], order[edge[1]]) for edge in list(cm)]
        )
        pm = generate_preset_pass_manager(
            optimization_level=3, 
            coupling_map=layout_cm, 
            basis_gates=['rz', 'rzz', 'cx', 'id', 'swap', 'u'],
        )
        init = pm.init
        # init.remove(3)
        pm.init = init
        pm.layout = None
        print('Transpiling accumulator')

        # compiled_circuit = circuit_to_dag(pm.run(dag_to_circuit(accumulator), callback=callback_func))
        compiled_circuit_dag = circuit_to_dag(pm.run(dag_to_circuit(temp_dag)))

        new_dag.compose(compiled_circuit_dag, qubits=order_bits)

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
                
        # TODO: write the logic for the subset aware layer building
        # Remove the Decompose pass as it will be handled here
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
        # TODO: strict subset interactions should be implemented at the same time as their superset interactions
        sub_layers = []
        current_sub_layer_index = 0
        while len(current_layer) > 0:
            current_sub_layer, remaining_gates = {}, {}
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

            # Not all gates that are applied at the ith swap layer can be applied at the same
            # time. We therefore greedily build sub-layers.
            # print(f'Layer {i}. Gates in layer: {current_layer.keys()}')
            # print(f'Unmapped gates in layer: {[indices for indices, _ in gate_layers.get(i, {}).items()]}')
            sub_layers = self._build_sub_layers(current_layer)
            # print(f'Layer {i}. Sub-layers: {len(sub_layers)}. Max interaction size: {max([len(key) for key in current_layer.keys()]) if len(current_layer.keys()) > 0 else 0}')

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
                        print(swap)
                        print(trivial_layout.get_physical_bits())
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
            
