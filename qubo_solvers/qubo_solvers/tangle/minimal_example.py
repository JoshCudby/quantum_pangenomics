import numpy as np
from dwave.system import LeapHybridSampler
from dimod.reference.samplers import SimulatedAnnealingSampler
from dimod import BQM

# sampler = LeapHybridSampler()
sampler = SimulatedAnnealingSampler()
qubo_matrix = np.array([
    [1, 0, 0],
    [0, 1, -1],
    [0, -1, -1]
])
bqm = BQM(qubo_matrix, 'BINARY')
###
# If using Leap Hybrid, time limit will be automatically computed to use one cycle of QPU use;
# else a custom limit can be applied via the time_limit parameter.
sampleset = sampler.sample(bqm)
best_sample = sampleset.first.sample
best_energy = sampleset.first.energy
print(f'Best Sample: {best_sample}')
print(f'Best energy: {best_energy}')