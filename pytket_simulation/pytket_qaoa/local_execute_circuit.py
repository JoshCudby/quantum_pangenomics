import numpy as np
import matplotlib.pyplot as plt
import sys
from pytket_qaoa.utils.qaoa_utils import qaoa_circuit, Q_to_Ising
from pytket_qaoa.utils.logging import get_logger
from pytket.transform import Transform
from pytket.circuit.display import get_circuit_renderer
from pytket.extensions.quantinuum import QuantinuumBackend, QuantinuumAPIOffline


logger = get_logger(__name__)
seed = 13
rng = np.random.default_rng(seed)


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
n_qubits = Q.shape[0]

h, J, offset = Q_to_Ising(Q, offset)
terms = list(J.items()) + [((key,), val) for key, val in h.items()]
circuit, parameters = qaoa_circuit(n_qubits, p, terms)
circuit = circuit.measure_all()


best_params = [4.0193,  2.8199,  1.3921, 0.9243, -0.3462, 2.9195,  0.9292, -0.0003]
optimised_symbol_map = {parameters[i]: best_params[i] for i in range(len(parameters))}
circuit.symbol_substitution(optimised_symbol_map)


logger.info(f'Num 1 qubit gates: {circuit.n_1qb_gates()}')
logger.info(f'Num 2 qubit gates: {circuit.n_2qb_gates()}')

api_offline = QuantinuumAPIOffline()
logger.info(f'Machines: {api_offline.get_machine_list()}')

backend = QuantinuumBackend(device_name="H2-2LE", api_handler=api_offline)
logger.info(backend.backend_info)
compiled_circuit = backend.get_compiled_circuit(circuit, optimisation_level=3)

logger.info(f'Num 1 qubit gates: {circuit.n_1qb_gates()}')
logger.info(f'Num 2 qubit gates: {circuit.n_2qb_gates()}')

handle = backend.process_circuit(compiled_circuit, n_shots=4000, seed=seed)

result = backend.get_result(handle)
# logger.info(result)
# shots = result._shots
# logger.info(shots.n_outcomes)
# logger.info(shots.width)
# logger.info(shots.to_readouts())

dist = result.get_empirical_distribution()
logger.info(dist)
optimal = [1,0,0,0,0,1,0,0,0,0,1,0]
dist_forward_opt = dist.condition(lambda x: all([x[i] == optimal[i] for i in range(len(optimal))]))

optimal.reverse()
dist_backward_opt = dist.condition(lambda x: all([x[i] == optimal[i] for i in range(len(optimal))]))
logger.info(dist_forward_opt)
logger.info(dist_backward_opt)
