from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import argparse
import numpy as np


# cacheblocking.26316  for a useful run with 12 qubits
# cacheblocking.26377  for a useful run with 4 qubits
# cacheblocking.26626  for a minimal run with 4 qubits

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--memory', type=int, default=4000)
parser.add_argument('-b', '--blocking', type=int, default=6)
parser.add_argument('-g', '--gpu', type=int, default=1)
args = parser.parse_args()

print(f'Blocking qubits: {args.blocking}, num gpus: {args.gpu}')

seed = 1
    

backend_options = dict(
    method='statevector',
    device='GPU',
    max_memory_mb=args.memory*0.9,
    cuStateVec_enable=True,
    precision='single'
)

backend_no_blocking = AerSimulator(**backend_options, blocking_enable=False, seed_simulator=seed)
backend_blocking = AerSimulator(**backend_options, blocking_enable=True, blocking_qubits=args.blocking, seed_simulator=seed)

N = 4

print(f'N={N}')

qc = QuantumCircuit(N)
for j in range(N-1):
    qc.h(j)

for j in range(N-1,0,-1):
    qc.mcx(list(range(j)), j)
qc.x(0)
    

t_qc_no_blocking = transpile(qc, backend_no_blocking, optimization_level=3, seed_transpiler=seed)
t_qc_no_blocking.save_statevector('final')
t_qc_no_blocking.measure_all()

t_qc_blocking = transpile(qc, backend_blocking, optimization_level=3, seed_transpiler=seed)
t_qc_blocking.save_statevector('final')
t_qc_blocking.measure_all()


expected_strings = ['0001','0010','0011','0100',
            '0101','0110','0111','1000']
expected_indexes = [1,2,3,4,5,6,7]

result_no_blocking = backend_no_blocking.run(t_qc_no_blocking, shots=10**7).result()
result_blocking = backend_blocking.run(t_qc_blocking, shots=10**7).result()

counts_no_blocking = result_no_blocking.get_counts()
counts_blocking = result_blocking.get_counts()

sv_no_blocking = result_no_blocking.data()['final'].data
sv_blocking = result_no_blocking.data()['final'].data

nz_no_blocking = np.nonzero(sv_no_blocking)
nz_blocking = np.nonzero(sv_blocking)

print('No blocking')
print(sv_no_blocking)
for index in nz_no_blocking[0]:
    if index not in expected_indexes:
        print(f'Rep: {np.binary_repr(index, qc.num_qubits)}. Amplitude: {np.abs(sv_no_blocking[index]) ** 2}')
for key in sorted(counts_no_blocking.keys()):
    if key not in expected_strings:
        print(f'key: {key}. count: {counts_no_blocking[key]}')


print('Blocking')
print(sv_blocking)
for index in nz_blocking[0]:
    if index not in expected_indexes:
        print(f'Rep: {np.binary_repr(index, qc.num_qubits)}. Amplitude: {np.abs(sv_no_blocking[index]) ** 2}')
for key in sorted(counts_blocking.keys()):
    if key not in expected_strings:
        print(f'key: {key}. count: {counts_blocking[key]}')
        
