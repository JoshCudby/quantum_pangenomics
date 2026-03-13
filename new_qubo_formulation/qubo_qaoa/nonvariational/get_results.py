from qiskit_ibm_runtime import QiskitRuntimeService
import pickle




service = QiskitRuntimeService(
    channel='ibm_quantum_platform',
    instance='crn:v1:bluemix:public:quantum-computing:us-east:a/cab32497708745b09a68cbfdf5f415aa:d4e63032-8000-4920-a6ed-26af87813cbe::'
)

res = {}
samples_dict = {}
all_samples = []

for job_id in [
    'd6nd5kk3pels73a1ft7g', # 0
    'd6nda869td6c73ao1ng0', # 2
    'd6ndevc3pels73a1g92g', # 4
    'd6ndjq0fh9oc73ep1rt0', # 6
    'd6ndoh43pels73a1gjgg' # 8
]:
    job = service.job(job_id)
    job_result = job.result()
    counts = job_result[0].data.c.get_counts()
    samples = []
    for idx, (sample, count) in enumerate(counts.items()):
        samples.extend(count * [sample])
    all_samples.append(samples)
    all_samples.append([])
    
samples_dict[(1,1)] = all_samples
res['samples_dict'] = samples_dict

filename = 'test_N4_W6'

append_str = f'.{filename}.error_mit.backendibm_boston.db0.63.dg0.16.shots400000.betaT0.15.eps0.05.alpha0.001'
with open(f'/lustre/scratch127/qpg/jc59/new_qubo_formulation/oriented/nonvariational/hardware/hardware{append_str}.pkl', 'wb') as f:
    pickle.dump(res, f)
