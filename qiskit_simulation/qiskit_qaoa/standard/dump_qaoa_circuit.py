"""Export a QAOA circuit to OpenQASM 3 format.

Builds a QAOAAnsatz from a QUBO data file, then serialises the unparameterised
(symbolic-parameter) circuit to an OpenQASM 3 string and writes it to a
``.qasm`` file.  The output is useful for sharing circuits, inspecting gate
counts, or importing into other quantum software stacks.

CLI usage::

    python dump_qaoa_circuit.py [data_file] [p]

Args:
    data_file (str): Path to the ``.npy`` QUBO data file
        (default: ``qubo_data_trivial.gfa.npy``).
    p (int): QAOA circuit depth / number of repetitions (default: 4).

Output:
    Writes ``qaoa_n<n>_p<p>_depth<depth>_size<size>.qasm`` to the qiskit
    output directory.
"""

import numpy as np
import sys
from qiskit_optimization import QuadraticProgram
from qiskit.circuit.library import QAOAAnsatz
from qiskit.qasm3 import dumps


if len(sys.argv) > 1:
    data_file = sys.argv[1]
else:
    data_file = '/lustre/scratch127/qpg/jc59/out/tangle/qubo_data_trivial.gfa.npy'


if len(sys.argv) > 2:
    p = int(sys.argv[2])
else:
    p = 4


data = np.load(data_file, allow_pickle=True)
Q, offset, T, N  = data
Q = np.triu(Q) * 2
Q -= np.triu(np.triu(Q).T) / 2
Q = Q / np.max(np.abs(Q))

mod = QuadraticProgram("QUBO test")
mod.binary_var_list(Q.shape[0])
mod.minimize(constant=offset, linear=None, quadratic=Q)
hamiltonian, offset = mod.to_ising()
hamiltonian = hamiltonian.sort(weight=True)


circuit = QAOAAnsatz(cost_operator=hamiltonian, reps=p, flatten=True)
circuit.measure_all()
n = circuit.num_qubits
size = circuit.size()
depth = circuit.depth()

f = open(f"/lustre/scratch127/qpg/jc59/out/qiskit/qaoa_n{n}_p{p}_depth{depth}_size{size}.qasm", "w")
qasm_str = dumps(circuit)
f.write(qasm_str)
f.close()