from qiskit.circuit.library import QuantumVolume
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt

sim = AerSimulator(method='statevector', device='GPU')

shots = 100
depth=10

time_thrust= []
time_cuStateVec= []
qubits_list = []

for qubits in range (15, 30):
    qubits_list.append(qubits)
    circuit = QuantumVolume(qubits, depth, seed=0)
    circuit.measure_all()

    pass_manager = generate_preset_pass_manager(optimization_level=3, backend=sim)
    compiled_circuit = pass_manager.run(circuit)

    result = sim.run(compiled_circuit,shots=shots,seed_simulator=12345,cuStateVec_enable=False).result()
    time_thrust.append(float(result.to_dict()['results'][0]['time_taken']))

    result_cuStateVec = sim.run(compiled_circuit,shots=shots,seed_simulator=12345,cuStateVec_enable=True).result()
    time_cuStateVec.append(float(result_cuStateVec.to_dict()['results'][0]['time_taken']))

fig, ax = plt.subplots()
ax.set_yscale("log")
ax.plot(qubits_list, time_thrust, marker="o", label='ThrustGPU')
ax.plot(qubits_list, time_cuStateVec, 'g', marker="x", label='cuStateVec')
ax.legend()
fig.savefig('/lustre/scratch127/qpg/jc59/out/qiskit/test_cuStateVec_no_fusion_threshold.png', format='png')
