import numpy as np
import sys
import pickle
from openqaoa.utilities import ground_state_hamiltonian
from openqaoa import QAOA
from openqaoa.backends.qaoa_device import create_device
from qiskit.quantum_info import Statevector
from openqaoa_tangle.utils.get_qubo import get_qubo
from openqaoa_tangle.utils.logging import get_logger

outdir = '/lustre/scratch127/qpg/jc59/out/openqaoa'
plotdir = '/nfs/users/nfs_j/jc59/quantumwork/pangenome/openqaoa_simulation/out'
logger = get_logger(__name__)

filename = sys.argv[1]
maxiter = 100
n_shots_min = 10
n_shots_max = 1000
p = 4

logger.info(filename)
logger.info(f'p: {p}')
logger.info(f'Max iter: {maxiter}')
logger.info(f'n shots min: {n_shots_min}')
logger.info(f'n shots max: {n_shots_max}')


ising_qubo = get_qubo(filename)
n = ising_qubo.n
# logger.info(ising_qubo.asdict())

ising_hamiltonian = ising_qubo.hamiltonian

if n < 15:
    # import the brute-force solver to obtain exact solution
    ising_energy, ising_configuration = ground_state_hamiltonian(ising_hamiltonian)
    logger.info(f"Ising Ground State energy: {ising_energy}, Solution: {ising_configuration}")


# initialize model with default configurations
q = QAOA()

# optionally configure the following properties of the model
q.set_backend_properties(use_gpu=True, n_shots=500)

# device
device = create_device(location='local', name='qiskit.shot_simulator')
q.set_device(device)

# circuit properties
q.set_circuit_properties(p=p, param_type='standard', init_type='ramp', mixer_hamiltonian='x')

q.set_backend_properties(prepend_state=None, append_state=None)

# classical optimizer properties
q.set_classical_optimizer(
    method='cans', 
    jac="param_shift", 
    maxiter=maxiter,
    optimizer_options=dict(
        stepsize=0.01, # stepsize < 2 /L, L < sum(terms in hamil) < nC2
        mu=0.95,
        b=0.001,        
        n_shots_min=n_shots_min,
        n_shots_max=n_shots_max,
    ),
    optimization_progress=True, cost_progress=True, parameter_log=True
)


q.compile(ising_qubo)
q.optimize()

opt_results = q.result
fig, ax = opt_results.plot_cost()
fig.savefig(f'{plotdir}/shots_qaoa_costs.{filename}.p{p}.maxiter{maxiter}.n_shots{n_shots_min}_{n_shots_max}.png')

logger.info(opt_results.optimized)
logger.info(opt_results.lowest_cost_bitstrings())

logger.info('Probability of uniform random')
logger.info(2 ** -n)

match filename:
    case 'trivial':    
        optimal = [1,0,0,0,0,1,0,0,0,0,1,0]
    case 'small_test':
        optimal = [1,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1]
    case 'test_N4_W5':
        optimal = [1,0,0,0,0,
                   0,1,0,0,0,
                   0,0,1,0,0,
                   1,0,0,0,0,
                   0,0,0,1,0,
                   0,0,0,0,1
                   ]
    case _:
        optimal = None

if optimal is not None:
    optimal_str = ''.join([str(x) for x in optimal])
    logger.info('Count of optimal')
    try:
        logger.info(opt_results.optimized['measurement_outcomes'][optimal_str])
    except KeyError:
        logger.info('Optimal was not sampled')


variational_params = q.optimizer.variational_params

#create the optimized QAOA circuit for qiskit backend
optimized_angles = opt_results.optimized['angles']
variational_params.update_from_raw(optimized_angles)
optimized_circuit = q.backend.qaoa_circuit(variational_params)
optimized_circuit.remove_final_measurements()

logger.info(variational_params)

statevector = Statevector(optimized_circuit)
if optimal is not None:
    optimal_idx = sum([optimal[i] * 2**(len(optimal)-1-i) for i in range(len(optimal))])
    logger.info('Probability of optimal')
    logger.info(np.abs(statevector[optimal_idx]) ** 2)

with open(f'{outdir}/shots_qaoa_results.{filename}.p{p}.maxiter{maxiter}.n_shots{n_shots_min}_{n_shots_max}.pkl', 'wb') as f:
    pickle.dump(opt_results, f)
    
fig = opt_results.plot_n_shots()[0]
fig.savefig(f'{plotdir}/shots_qaoa_num_shots.{filename}.p{p}.maxiter{maxiter}.n_shots{n_shots_min}_{n_shots_max}.png')