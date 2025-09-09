
import re
from qiskit import QuantumCircuit
from qiskit.transpiler import Layout
from qiskit.transpiler.passes import LayoutTransformation
from qiskit.converters import dag_to_circuit, circuit_to_dag


def swap_between_circuit_layouts(index: int, compiled_circuits: dict[int, QuantumCircuit], edge_maps: dict[int, dict[int, int]], coupling_map):
    num_qubits = list(compiled_circuits.values())[0].num_qubits
    if index < 0:
        return QuantumCircuit(num_qubits)
    from_layout = Layout({i: compiled_circuits[index].qubits[edge_maps[index][i]] for i in range(num_qubits)})
    for instruction in compiled_circuits[index].data:
        if instruction.operation.name == 'swap':
            qubits_str = str(instruction.qubits)
            matches = re.findall('index=([0-9]+)', qubits_str)
            if len(matches) == 2:
                from_layout.swap(compiled_circuits[0].qubits[int(matches[0])], compiled_circuits[0].qubits[int(matches[1])])
            else:
                raise Exception('Did not find 2 swap indices')
    to_layout = Layout({i: compiled_circuits[index+1].qubits[edge_maps[index+1][i]] for i in range(num_qubits)})
    transformation_pass = LayoutTransformation(coupling_map, from_layout, to_layout)
    swap_qc = QuantumCircuit(num_qubits)
    swap_qc = dag_to_circuit(transformation_pass.run(circuit_to_dag(swap_qc)))
    return swap_qc