import numpy as np
import sys
from qiskit_optimization import QuadraticProgram
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer import AerSimulator, AerError
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_qaoa.utils.qaoa_utils import optimize_qaoa_parameters
from qiskit_qaoa.utils.sample_utils import sample_optimized_circuit
from qiskit_qaoa.utils.string_utils import bitstring_to_energy

seed = 10

np.random.seed(seed)
rng = np.random.default_rng(seed)

if len(sys.argv) > 1:
    data_file = sys.argv[1]
else:
    data_file = '/lustre/scratch127/qpg/jc59/out/tangle/qubo_data_trivial.gfa.npy'

if len(sys.argv) > 2:
    method = str(sys.argv[2])
else:
    method = 'automatic'

if len(sys.argv) > 3:
    use_gpu = int(sys.argv[3])
else:
    use_gpu = False

data = np.load(data_file, allow_pickle=True)
Q, offset, T, N  = data
Q = np.triu(Q) * 2
Q -= np.triu(np.triu(Q).T) / 2

mod = QuadraticProgram("QUBO test")
mod.binary_var_list(Q.shape[0])
mod.minimize(constant=offset, linear=None, quadratic=Q)
op, offset = mod.to_ising()
op = op.sort(weight=True)


p = 4
circuit = QAOAAnsatz(cost_operator=op, reps=p, flatten=True)
circuit.measure_all()
print(circuit.num_qubits)

try:
    ideal_aer = AerSimulator(
        method=method,
        matrix_product_state_max_bond_dimension=5,
        device='GPU' if use_gpu else 'CPU',
        blocking_enable=True, blocking_qubits=20
    )
except AerError as error:
    print(error)

print(ideal_aer.available_devices())

# Create pass manager for transpilation
ideal_pm = generate_preset_pass_manager(optimization_level=3, backend=ideal_aer)
ideal_circuit = ideal_pm.run(circuit)
# ideal_circuit.measure_all()

# gamma, beta = rng.random((2, p))
gamma, beta = np.zeros((2, p))
init_params = [*gamma, *beta]

parameter_binding = {
    ideal_circuit.parameters[i]: [init_params[i]] for i in range(len(init_params))
}

opt_result = optimize_qaoa_parameters(
    ideal_aer,
    init_params,
    ideal_circuit,
    op,
    p,
    estimator_shots=1e5
)
optimized_params = [float(param) for param in opt_result.x] 

sample = sample_optimized_circuit(
    ideal_aer,
    ideal_circuit,
    optimized_params
)

keys = list(sample.keys())
values = list(sample.values())
most_likely_bitstring = [x for x in keys[np.argmax(np.abs(values))]]
most_likely_bitstring.reverse()


print(f'Result bitstring: {most_likely_bitstring}')
print(f'Prob of most likely: {np.max(np.abs(values))}')
print(f'Most likely energy: {bitstring_to_energy(most_likely_bitstring, op)}')

print(f'Uniform random prob: {2 ** -Q.shape[0]}')

if data_file == "/lustre/scratch127/qpg/jc59/out/tangle/qubo_data_trivial.gfa.npy":
    optimal = [1,0,0,0,0,1,0,0,0,0,1,0]
    optimal.reverse()
    print(f'Optimal bitstring: {optimal}')
    print(f'Optimal energy: {bitstring_to_energy(optimal, op)}')
    print(f'Prob of optimal: {sample["".join([str(x) for x in optimal])]}')
elif data_file == "/lustre/scratch127/qpg/jc59/out/tangle/qubo_data_small_test.gfa.npy":
    optimal = [
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        1,0,0,0,
        0,1,0,0,
        0,0,0,1
    ]
    optimal.reverse()
    print(f'Optimal bitstring: {optimal}')
    print(f'Optimal energy: {bitstring_to_energy(optimal, op)}')
    print(f'Prob of optimal: {sample["".join([str(x) for x in optimal])]}')
elif data_file == "/lustre/scratch127/qpg/jc59/out/tangle/qubo_data_test.gfa.npy":
    optimal = [
        1,0,0,0,0,0,
        0,1,0,0,0,0,
        0,0,1,0,0,0,
        0,0,0,1,0,0,
        0,1,0,0,0,0,
        0,0,1,0,0,0,
        0,0,0,0,1,0,
        0,0,0,0,0,1
    ]
    optimal.reverse()
    print(f'Optimal bitstring: {optimal}')
    print(f'Optimal energy: {bitstring_to_energy(optimal, op)}')
    print(f'Prob of optimal: {sample["".join([str(x) for x in optimal])]}')

    optimal = [
        1,0,0,0,0,0,
        0,1,0,0,0,0,
        0,0,1,0,0,0,
        0,1,0,0,0,0,
        0,0,0,1,0,0,
        0,0,1,0,0,0,
        0,0,0,0,1,0,
        0,0,0,0,0,1
    ]
    optimal.reverse()
    print(f'Optimal bitstring: {optimal}')
    print(f'Optimal energy: {bitstring_to_energy(optimal, op)}')
    print(f'Prob of optimal: {sample["".join([str(x) for x in optimal])]}')
