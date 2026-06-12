import sys
import pickle
import matplotlib.pyplot as plt
from openqaoa.utilities import ground_state_hamiltonian
from openqaoa import RQAOA
from openqaoa.backends.qaoa_device import create_device
from openqaoa_tangle.utils.get_qubo import get_qubo
from openqaoa_tangle.utils.logging import get_logger

outdir = '/lustre/scratch127/qpg/jc59/out/openqaoa'
plotdir = '/nfs/users/nfs_j/jc59/quantumwork/pangenome/openqaoa_simulation/out'
logger = get_logger(__name__)

filename = sys.argv[1]
p = 2
max_iter = 100
n_max = 3

logger.info(filename)
logger.info(p)


ising_qubo = get_qubo(filename)
n = ising_qubo.n
# logger.info(ising_qubo.asdict())

ising_hamiltonian = ising_qubo.hamiltonian

if n < 15:
    # import the brute-force solver to obtain exact solution
    ising_energy, ising_configuration = ground_state_hamiltonian(ising_hamiltonian)
    logger.info(f"Ising Ground State energy: {ising_energy}, Solution: {ising_configuration}")


r =  RQAOA()
r.set_backend_properties(use_gpu=True)

r.set_rqaoa_parameters(rqaoa_type='adaptive', n_max=n_max, n_cutoff=10)

r.set_circuit_properties(p=p, param_type='standard', init_type='ramp', mixer_hamiltonian='x')

device = create_device(location='local', name='qiskit.statevector_simulator' if n > 20 else 'vectorized')
r.set_device(device)

r.set_classical_optimizer(method='cobyla', maxiter=max_iter)


r.compile(ising_qubo)

r.optimize(
    dump=True,
    dump_options={"file_name":"rqaoa_results", "file_path": outdir, "prepend_id": True}
)

# Extract results
opt_results = r.result
num_steps = opt_results['number_steps']
fig, axs = plt.subplots(num_steps)
for i in range(num_steps):
    res = opt_results.get_qaoa_results(i)
    res.plot_cost(ax=axs[i])
fig.savefig(f'/nfs/users/nfs_j/jc59/quantumwork/pangenome/openqaoa_simulation/out/rqaoa_costs.{filename}.p{p}.n_max{n_max}.maxiter{max_iter}.png')

logger.info(opt_results['solution'])
with open(f'/lustre/scratch127/qpg/jc59/out/openqaoa/rqaoa_results.{filename}.p{p}.n_max{n_max}.maxiter{max_iter}.pkl', 'wb') as f:
    pickle.dump(opt_results, f)