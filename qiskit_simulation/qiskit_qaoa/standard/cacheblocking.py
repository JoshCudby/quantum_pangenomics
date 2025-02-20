from qiskit_aer import AerSimulator
from qiskit.circuit.library import QuantumVolume
from qiskit import transpile
import sys

qubits = int(sys.argv[1])

blocking_qubits = 23
max_memory_mb = 512000

# Blocking qubits = "chunk size" from heterogenous sim paper? So actually a trade-off in desired size
print('Blocking qubits formula in MiB assuming sizeof(complex) == 16')
print(16 * 2 ** (blocking_qubits+4) / (1024 ** 2))
print('Gpu memory: 80000MiB')

print('Circuit required memory')
print(16 * 2 ** qubits / (1024 ** 2))

print('Available memory with 80000MiB per 1 of 4 gpus')
print(max_memory_mb + 80000 * 4)

sim = AerSimulator(method='statevector', device='GPU', blocking_enable=True, blocking_qubits=blocking_qubits, max_memory_mb=max_memory_mb)

print(sim.available_devices())
circ = transpile(QuantumVolume(qubits, 10, seed = 0))
circ.measure_all()
result = sim.run(circ, shots=100, blocking_enable=True, blocking_qubits=blocking_qubits).result()

try:
    print('Result.metadata')
    print(result.metadata)
except:
    pass  
try:
    print('Result.to_dict()')
    print(result.to_dict())
    print('Result.to_dict()[metadata]')
    print(result.to_dict()['metadata'])
    print(result.to_dict()['results'][0]['metadata']['cacheblocking'])
except:
    pass
try:
    print('Result.metadata.cacheblocking')
    print(result.metadata.cacheblocking)
except:
    pass  
try:
    print('Result.cacheblocking')
    print(result.cacheblocking)
except:
    pass  

print(result)

# 263093: blocking=30, max mem 205000, mem 400000, 4 gpu * 81000 = 324000, required = 524288
# 263139: blocking=28 (), max mem 205000, mem 400000, 4 gpu * 81000 = 324000, required = 524288
# sizeof(complex) * 