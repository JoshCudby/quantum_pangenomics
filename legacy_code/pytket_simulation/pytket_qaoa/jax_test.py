from jax import numpy as jnp
import jax
from jax.sharding import PartitionSpec as P
# import qujax
# from pytket.extensions.qujax import tk_to_qujax
# from pytket_qaoa.utils.qaoa_utils import qaoa_circuit, Q_to_Ising
from pytket_qaoa.utils.logging import get_logger

logger = get_logger(__name__)

logger.info(jax.devices())
logger.info(jax.devices()[0].memory_stats())

mesh = jax.make_mesh((1, 3), ('x', 'y'))
sharding = jax.sharding.NamedSharding(mesh, P('x', 'y'))

arr = jnp.ones((2**10, 3*2**20), device=sharding)
logger.info(arr.sharding)
logger.info(jax.devices()[0].memory_stats())


# data_file = '/lustre/scratch127/qpg/jc59/out/tangle/qubo_data_trivial.gfa.npy'
# data = jnp.load(data_file, allow_pickle=True)
# Q, offset, T, N  = data
# Q = jnp.triu(Q) * 2
# Q -= jnp.triu(jnp.triu(Q).T) / 2
# n_qubits = Q.shape[0]

# h, J, offset = Q_to_Ising(Q, offset)
# terms = list(J.items()) + [((key,), val) for key, val in h.items()]
# hamiltonian_qubit_inds = list(J.keys()) + [(key,) for key in h.keys()]
# coefficients = list(J.values()) + list(h.values())
# hamiltonian_gates = [
#     ['Z'] * len(x) for x in hamiltonian_qubit_inds
# ]

# logger.info(f'Num qubits: {n_qubits}')
# logger.info(f'Gates:\t {hamiltonian_gates}')
# logger.info(f'Qubits:\t {hamiltonian_qubit_inds}')
# logger.info(f'Coefficients:\t {coefficients}')
# logger.info(terms)

# circuit, parameters = qaoa_circuit(n_qubits, p, terms)

# symbol_map = {parameters[i]: i for i in range(len(parameters))}

# param_to_st = tk_to_qujax(circuit, symbol_map=symbol_map)
# st_to_expectation = qujax.get_statetensor_to_expectation_func(
#     hamiltonian_gates, hamiltonian_qubit_inds, coefficients
# )

# def param_to_expectation(param): 
#     return st_to_expectation(param_to_st(param))


# cost_and_grad = jit(value_and_grad(param_to_expectation))