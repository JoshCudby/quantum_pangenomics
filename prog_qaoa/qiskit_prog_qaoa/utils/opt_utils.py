import numpy as np
from time import time
import networkx as nx

from scipy.optimize import OptimizeResult

from qiskit import QuantumCircuit
from qiskit_aer.primitives import SamplerV2 as Sampler

from qiskit_prog_qaoa.utils.logging import get_logger

logger = get_logger(__name__)


def soln_to_path(soln, n, T, graph):
    ceil_log_n2 = int(np.ceil(np.log2(n+2)))

    path = []
    nodes = list(graph.nodes)
    for t in range(T):
        x_bin = soln[t * ceil_log_n2: (t+1) * ceil_log_n2]
        x_int = sum(2 ** (ceil_log_n2-i-1) * int(x_bin[i]) for i in range(ceil_log_n2))
        if x_int < n+1:
            path.append(nodes[x_int -1])
        elif x_int == n+1:
            path.append('end')
        else:
            path.append('invalid node')
    return path



def cost_function(sample: str, n, T, graph: nx.Graph, lamda) -> float:
    ceil_log_n2 = int(np.ceil(np.log2(n+2)))

    nodes = list(graph.nodes)
    cost = 0
    x = []
    counts = {}
    for t in range(T):
        x_int = sum(2 ** (ceil_log_n2-i-1) * int(sample[t * ceil_log_n2 + i]) for i in range(ceil_log_n2))
        x.append(x_int)
        counts[x_int] = counts.get(x_int, 0) + 1

    for t in range(T-1):
        if x[t] > n+1:
            logger.error(f'Sampled an invalid node! Sample ints: {x}. Sample: {sample}.')
            cost += 10000 # Should never happen
        elif x[t] == n+1:
            if not x[t+1] == n+1:
                cost += lamda
        else:
            if x[t+1] > n+1:
                pass
            elif not x[t+1] == n+1 and (nodes[x[t]-1], nodes[x[t+1]-1]) not in graph.edges:
                cost += lamda

    for i in range(1, n+1):
        cost += (counts.get(i, 0) - graph.nodes[nodes[i-1]]["weight"]) ** 2

    if cost == 0.0:
        logger.info(f'Sampled optimum: {sample}. Path: {soln_to_path(sample, n, T, graph)}')
    return cost
    

def cvar(counts, n, T, graph, lamda, alpha=1.0):
    evals = [cost_function(key, n, T, graph, lamda) for key in counts.keys()]
    energies = [count * [evals[idx]] for idx, count in enumerate(counts.values())]
    energies = sorted([x for xs in energies for x in xs])
    # if energies[0] == 0:
    #     return -1
    end_idx = int(min(alpha,1) * len(energies))
    return np.sum(energies[0:end_idx]) / end_idx


def objective(x: np.ndarray, n, T, graph, lamda, shots, history: list, circuit: QuantumCircuit, sampler: Sampler):
    start = time()
    assigned_circuit = circuit.assign_parameters(x, inplace=False)
    sampler_job = sampler.run([assigned_circuit], shots=shots)
    try:
        sampler_result = sampler_job.result()
        
        counts = sampler_result[0].data.c.get_counts()
        sampling_time = time() - start
        start = time()
        total_energy = cvar(counts, n, T, graph, lamda, alpha=0.05)
        
        classical_post_process_time = time() - start
        
        history.append((sampling_time, total_energy, x.tolist(), counts, classical_post_process_time))
        return total_energy
    except Exception as e:
        logger.error(e)
        logger.error(sampler_job.result())
        logger.error(sampler_job.result()[0].data)
        logger.error(sampler_job.result()[0].data.c)


def callback(intermediate_result: OptimizeResult):
    logger.info(f'Current params: {intermediate_result.x}. Current func value: {intermediate_result.fun}')
    if intermediate_result.fun == -1:
        raise StopIteration
    

class TerminationChecker:
    def __init__(self):
        pass

    def __call__(self, nfev, parameters, value, stepsize, accepted) -> bool:
        logger.info(f'Current params: {parameters}. Current func value: {value}')
        if value == 0:
            return True
        return False
    