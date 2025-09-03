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
            logger.warning(f'Multi-qubit operator that we cannot implement but is in decomposition: {qubit}') 
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
                    


