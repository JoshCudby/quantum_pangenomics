"""Smoke test for QAOAAnsatz parameter binding conventions.

Constructs a small random 2-qubit QUBO and a depth-2 QAOAAnsatz circuit,
then verifies that Qiskit's two parameter-binding APIs produce equivalent
circuits:

* **List-style binding** — ``circuit.assign_parameters(init_params)``
* **Dict-style binding** — ``circuit.assign_parameters({param: value, ...})``

Both bound circuits are drawn and saved to the qiskit output directory so that
parameter ordering can be inspected visually.  Also prints ``circuit.parameters``
and ``circuit.ordered_parameters`` to confirm ordering consistency.
"""

from qiskit.circuit.library import QAOAAnsatz
from qiskit_optimization import QuadraticProgram
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()

Q = rng.random((2, 2))
offset= 0
p = 2
mod = QuadraticProgram("QUBO test")
mod.binary_var_list(Q.shape[0])
mod.minimize(constant=offset, linear=None, quadratic=Q)
hamiltonian, offset = mod.to_ising()
hamiltonian = hamiltonian.sort(weight=True)

circuit = QAOAAnsatz(cost_operator=hamiltonian, reps=p, flatten=True)

gamma_bounds = (-np.pi, np.pi)
beta_bounds = (-np.pi/2, np.pi/2)

init_params = rng.random((2*p,)) \
    * np.array([beta_bounds[1] - beta_bounds[0]] * p + [gamma_bounds[1] - gamma_bounds[0]] * p) \
    + np.array([beta_bounds[0]] * p + [gamma_bounds[0]] * p)

parameter_binding = {
    circuit.parameters[i]: init_params[i] for i in range(len(init_params))
}
print(circuit.parameters)
print(circuit.ordered_parameters)

print(init_params)

param_circuit = circuit.assign_parameters(init_params)
bound_circuit = circuit.assign_parameters(parameter_binding)

param_circuit.draw(filename='/lustre/scratch127/qpg/jc59/out/qiskit/qaoa_list_circuit')
bound_circuit.draw(filename='/lustre/scratch127/qpg/jc59/out/qiskit/qaoa_dict_circuit')

