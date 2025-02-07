from pytket_qaoa.utils.logging import get_logger
import qnexus as qnx
import datetime

jobname_suffix = datetime.datetime.now().strftime('%Y_%m_%d-%H-%M-%S')

project = qnx.projects.get_or_create('QAOA_nexus_2025-02-06 17:17:06.612526')
qnx.context.set_active_project(project)
config = qnx.QuantinuumConfig(device_name='H1-1LE')


logger = get_logger(__name__)
circuits = qnx.circuits.get_all('Optimized_QAOA_Circuit')
latest_circuit_ref = circuits.list()[-1]
circuit = latest_circuit_ref.download_circuit()
circuit.measure_all()

ref = qnx.circuits.upload(circuit=circuit, name=latest_circuit_ref.annotations.name + 'measured')


logger.info('Starting nexus execute job')
shots=500
ref_execute_job = qnx.start_execute_job(
    circuits=[ref],
    n_shots=[shots],
    backend_config=config,
    name=f'execution-job-shots{shots}-{jobname_suffix}'
)
qnx.jobs.wait_for(ref_execute_job)
ref_result = qnx.jobs.results(ref_execute_job)[0]
backend_result = ref_result.download_result()
distribution = backend_result.get_empirical_distribution()
logger.info(distribution)