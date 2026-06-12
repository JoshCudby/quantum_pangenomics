import sys
import pickle
import numpy as np
from openqaoa import QAOA
from openqaoa.utilities import bitstring_energy
from openqaoa.backends.qaoa_device import create_device
from openqaoa_tangle.utils.get_qubo import get_qubo
from openqaoa_tangle.utils.logging import get_logger

outdir = '/lustre/scratch127/qpg/jc59/out/openqaoa'
plotdir = '/nfs/users/nfs_j/jc59/quantumwork/pangenome/openqaoa_simulation/out'
logger = get_logger(__name__)

filename = sys.argv[1]
maxiter = 100
q = 3

logger.info(filename)


ising_qubo = get_qubo(filename)
n = ising_qubo.n

device = create_device(location='local', name='qiskit.statevector_simulator' if n > 20 else 'vectorized')

# Initialize with "standard" Fourier parameters
# u = np.array([0.35])
# v = np.array([0.35])

# Initialize with "warmstart" Fourier parameters for small_test
u = np.array([0.17657293, 0.00071805, 0.00039475])
v = np.array([0.10691546, 0.00043302, 0.00098494])
p = 5

while p < 11:
    logger.info(f'p: {p}')
    qaoa = QAOA()
    qaoa.set_backend_properties(use_gpu=True)
    qaoa.set_device(device)
    qaoa.set_classical_optimizer(method='nelder-mead', maxiter=maxiter, tol=0.01,
                            optimization_progress=True, cost_progress=True, parameter_log=True)
    qaoa.set_circuit_properties(p=p, q=min(p, q), param_type='fourier', init_type='custom', mixer_hamiltonian='x',
                                variational_params_dict={'q': min(p, q), 'u': u, 'v': v})
    qaoa.compile(ising_qubo)
    qaoa.optimize()

    opt_results = qaoa.result

    logger.info(opt_results.optimized)
    with open(f'{outdir}/qaoa_results.{filename}.p{p}.q{q}.maxiter{maxiter}.pkl', 'wb') as f:
        pickle.dump(opt_results, f)
    
    variational_params = qaoa.optimizer.variational_params
    optimized_angles = opt_results.optimized['angles']
    variational_params.update_from_raw(optimized_angles)

    
    outcomes = opt_results.optimized['measurement_outcomes']
    probabilities = np.real(np.conjugate(outcomes) * outcomes)
    top_100_indices = np.argpartition(probabilities, -100)[-100:]
    solution_bitstring = [np.binary_repr(x, n)[::-1] for x in top_100_indices]
    energies = [
        bitstring_energy(opt_results.cost_hamiltonian, bitstring)
        for bitstring in solution_bitstring
    ]
    
    min_e = np.min(energies)
    argmin_e = np.argmin(energies)
    opt_found = min_e < 1e-6 and probabilities[top_100_indices[argmin_e]] > 5e-3
    logger.info(f'Min energy: {np.min(energies)}')
    logger.info(f'Solution: {solution_bitstring[argmin_e]}')
    logger.info(f'Probability: {probabilities[top_100_indices[argmin_e]]}')
    if opt_found:
        break
    
    u = variational_params.u
    if len(u) < q:
        u = np.append(u, 0)
    v = variational_params.v
    if len(v) < q:
        v = np.append(v, 0)
    p += 1

logger.info(variational_params)

