from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


def parameterise_circuit(qc: QuantumCircuit, parameter: Parameter) -> QuantumCircuit:
    clone = QuantumCircuit(qc.num_qubits)

    for g in qc:
        if len(g.operation.params) == 0:
            clone.append(g)
        elif len(g.operation.params) > 1:
            raise Exception('Only single parameter circuits allowed.')
        else:
            op = g.operation.copy()
            op.params = [g.operation.params[0] * parameter]
            clone.append(op, g.qubits)
    return clone