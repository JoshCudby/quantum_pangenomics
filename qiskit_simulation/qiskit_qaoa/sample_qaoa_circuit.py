import numpy as np
import sys
from qiskit.circuit.library import QAOAAnsatz
from qiskit_optimization import QuadraticProgram
from qiskit_aer.primitives import SamplerV2 as Sampler
from qiskit_qaoa.utils.sample_utils import sample_optimized_circuit, get_optimized_circuit_probabilities
from qiskit_qaoa.utils.string_utils import bitstring_to_energy, print_optimal_solution_properties
from qiskit_qaoa.utils.logging import get_logger

logger = get_logger(__name__)


if len(sys.argv) > 1:
    data_file = sys.argv[1]
else:
    data_file = '/lustre/scratch127/qpg/jc59/out/tangle/qubo_data_trivial.gfa.npy'

if len(sys.argv) > 2:
    p = int(sys.argv[2])
else:
    p = 4

if len(sys.argv) > 3:
    use_gpu = int(sys.argv[3])
else:
    use_gpu = False

if len(sys.argv) > 4:
    seed = int(sys.argv[4])
else:
    seed = 0


data = np.load(data_file, allow_pickle=True)
Q, offset, T, N  = data
Q = np.triu(Q) * 2
Q -= np.triu(np.triu(Q).T) / 2

normalisation = np.max(np.abs(Q))
Q = Q / normalisation
offset = offset / normalisation

# TODO: better data save/load
to_load = f'/lustre/scratch127/qpg/jc59/out/qiskit/qaoa_params_n{Q.shape[0]}_p{p}_seed{seed}.npy'
optimized_params = np.load(to_load)


mod = QuadraticProgram("QUBO test")
mod.binary_var_list(Q.shape[0])
mod.minimize(constant=offset, linear=None, quadratic=Q)
hamiltonian, offset = mod.to_ising()
hamiltonian = hamiltonian.sort(weight=True)


circuit = QAOAAnsatz(cost_operator=hamiltonian, reps=p, flatten=True)

# batched_shots_gpu=True
# batched_shots_gpu_max_qubits=30
# logger.info(f'Batched shots GPU: {batched_shots_gpu}, max qubits: {batched_shots_gpu_max_qubits}')
sampler = Sampler(
    options=dict(backend_options=dict(
        device='GPU' if use_gpu else 'CPU'
        # batched_shots_gpu=batched_shots_gpu, batched_shots_gpu_max_qubits=batched_shots_gpu_max_qubits
        ))
)
sample = sample_optimized_circuit(
    circuit,
    optimized_params,
    sampler
)

keys = list(sample.keys())
values = list(sample.values())
most_likely_bitstring = [int(x) for x in keys[np.argmax(np.abs(values))]]
most_likely_bitstring.reverse()

logger.info(f'Model offset: {offset}')

logger.info(f'Most likely bitstring: {most_likely_bitstring}')
logger.info(f'Prob of most likely: {np.max(np.abs(values))}')
logger.info(f'Most likely cost: {bitstring_to_energy(most_likely_bitstring, hamiltonian) + offset}')

logger.info(f'Uniform random prob: {2 ** -Q.shape[0]}')

if data_file == "/lustre/scratch127/qpg/jc59/out/tangle/qubo_data_trivial.gfa.npy":
    optimal = [1,0,0,0,0,1,0,0,0,0,1,0]
    print_optimal_solution_properties(optimal, hamiltonian, sample, offset)
elif data_file == "/lustre/scratch127/qpg/jc59/out/tangle/qubo_data_small_test.gfa.npy":
    optimal = [
        1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        1,0,0,0,
        0,1,0,0,
        0,0,0,1
    ]
    print_optimal_solution_properties(optimal, hamiltonian, sample, offset)
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
    print_optimal_solution_properties(optimal, hamiltonian, sample, offset)

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
    print_optimal_solution_properties(optimal, hamiltonian, sample, offset)
elif data_file == "/lustre/scratch127/qpg/jc59/out/tangle/qubo_data_test_N4_W5.gfa.npy":
    optimal = [
        1,0,0,0,0,
        0,1,0,0,0,
        0,0,1,0,0,
        1,0,0,0,0,
        0,0,0,1,0,
        0,0,0,0,1
    ]
    print_optimal_solution_properties(optimal, hamiltonian, sample, offset)



probs = get_optimized_circuit_probabilities(circuit, optimized_params)
logger.info(probs)
