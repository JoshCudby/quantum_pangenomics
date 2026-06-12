"""Qubit layout transformation utilities.

Provides helpers for computing the SWAP circuit needed to transition between
consecutive qubit layouts when stitching together separately compiled QAOA
cost layers.
"""

import re
from qiskit import QuantumCircuit
from qiskit.transpiler import Layout
from qiskit.transpiler.passes import LayoutTransformation
from qiskit.converters import dag_to_circuit, circuit_to_dag


def swap_between_circuit_layouts(index: int, compiled_circuits: dict[int, QuantumCircuit], layouts: dict[int, Layout], coupling_map):
    """Generate the SWAP circuit that bridges the output layout of circuit ``index`` to the input layout of circuit ``index+1``.

    Tracks SWAP gates applied within ``compiled_circuits[index]`` to determine
    the effective output layout, then uses ``LayoutTransformation`` to compute
    the SWAP sequence needed to reach ``layouts[index+1]``.

    Args:
        index: Index into ``compiled_circuits`` and ``layouts`` of the circuit
            whose output layout is to be connected.  Passing ``-1`` returns an
            empty circuit (useful as a boundary condition).
        compiled_circuits: A dict mapping integer indices to compiled
            ``QuantumCircuit`` objects that may contain ``swap`` gates.
        layouts: A dict mapping integer indices to the initial ``Layout`` of
            each compiled circuit.
        coupling_map: A Qiskit ``CouplingMap`` used by ``LayoutTransformation``
            to determine valid SWAP placements.

    Returns:
        A ``QuantumCircuit`` containing only SWAP gates that transforms the
        qubit permutation from the end of ``compiled_circuits[index]`` to the
        start of ``compiled_circuits[index+1]``.

    Raises:
        Exception: If a SWAP gate in the compiled circuit does not contain
            exactly two qubit indices.
    """
    num_qubits = list(compiled_circuits.values())[0].num_qubits
    if index < 0:
        return QuantumCircuit(num_qubits)
    from_layout = layouts[index].copy()
    for instruction in compiled_circuits[index].data:
        if instruction.operation.name == 'swap':
            qubits_str = str(instruction.qubits)
            matches = re.findall('index=([0-9]+)', qubits_str)
            if len(matches) == 2:
                from_layout.swap(int(matches[0]), int(matches[1]))
            else:
                raise Exception('Did not find 2 swap indices')
    to_layout = layouts[index+1].copy()
    transformation_pass = LayoutTransformation(coupling_map, from_layout, to_layout)
    swap_qc = QuantumCircuit(num_qubits)
    swap_qc = dag_to_circuit(transformation_pass.run(circuit_to_dag(swap_qc)))
    return swap_qc