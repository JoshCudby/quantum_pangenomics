import quimb as qu
import quimb.tensor as qtn
import sys
import cotengra as ctg

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    

opt = ctg.ReusableHyperOptimizer(
    methods=['greedy'],
    reconf_opts={}, 
    max_repeats=32,
    max_time="rate:1e6",
    parallel=True,
    # use the following for persistently cached paths
    # directory=True,
)

import networkx as nx

reg = 3
n = 54
seed = 666
G = nx.random_regular_graph(reg, n, seed=seed)

terms = {(i, j): 1 for i, j in G.edges}

p = 4
gammas = qu.randn(p)
betas = qu.randn(p)
circ_ex = qtn.circ_qaoa(terms, p, gammas, betas)

import tqdm

ZZ = qu.pauli('Z') & qu.pauli('Z')

local_exp_rehs = [
    circ_ex.local_expectation_rehearse(weight * ZZ, edge, optimize=opt)
    for edge, weight in tqdm.tqdm(list(terms.items()))
]
eprint([rehs['W'] for rehs in local_exp_rehs])
eprint([rehs['tree'].contraction_cost(log=10) for rehs in local_exp_rehs])


def energy(x):
    p = len(x) // 2
    gammas = x[:p]
    betas = x[p:]
    circ = qtn.circ_qaoa(terms, p, gammas, betas)

    ZZ = qu.pauli('Z') & qu.pauli('Z')

    ens = [
        circ.local_expectation(weight * ZZ, edge, optimize=opt, backend="jax")
        for edge, weight in terms.items()
    ]
    
    return sum(ens).real

from skopt import Optimizer

eps = 1e-6
bounds = (
    [(0.0        + eps, qu.pi / 2 - eps)] * p + 
    [(-qu.pi / 4 + eps, qu.pi / 4 - eps)] * p
)

bopt = Optimizer(bounds)

for i in tqdm.trange(100):
    x = bopt.ask()
    res = bopt.tell(x, energy(x))
    
print(res)