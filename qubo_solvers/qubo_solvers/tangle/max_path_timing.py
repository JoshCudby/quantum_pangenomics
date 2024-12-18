import sys
import numpy as np
from datetime import datetime
from qubo_solvers.definitions import Solver
from qubo_solvers.tangle.utils.setup_utils import setup
from qubo_solvers.tangle.utils.sampling_utils import dwave_sample_qubo, mqlib_sample_qubo, gurobi_sample_qubo, validate_path, qubo_vars_to_path
import timeit

print(timeit.timeit(
    stmt='setup(*sys.argv)',
    setup='from qubo_solvers.tangle.utils.setup_utils import setup',
    number=10
))


stp = """
from qubo_solvers.tangle.utils.sampling_utils import qubo_vars_to_path
import numpy as np
from qubo_solvers.tangle.utils.setup_utils import setup
qubo_data = np.load('/lustre/scratch127/qpg/jc59/out/tangle/mqlib_ddDapMeze1.MT.k301.utg.final.gfa_13122024_1319.npy', allow_pickle=True)
solution = qubo_data[0]
filepath, filename, tangle_out_dir, graph, time_limit, Q, offset, T_max, V, solver = setup(*sys.argv)
"""
print(timeit.timeit(
    stmt='qubo_vars_to_path(solution, graph)',
    setup=stp,
    number=10
))

# u21+,u23+,u25+,u27+,u39-,u30-,u29-,u24-,u23+,u25+,u0-,u2+,u3+,u6+,u10-,u37+,u5-,u3-,u2-,
# u1+,u11-,u12+,u14+,u38-,u16+,u35+,u32-,u33+,u34+,u40-,u7-,u3-,u2-,u1+,u36-,u35-,u20-,u18-,
# u17+,u39+,u26-,u23-,u21-,u19-,u18-,u15-,u38+,u13-,u12-,u9-,u37+,u8+,u29+,u31+,u40-,u4-,u3+,u6+,u28+,u33-,u22-
best_path_nodes = ["u21+","u23+","u25+","u27+","u39-","u30-","u29-","u24-","u23+","u25+","u0-","u2+","u3+","u6+","u10-","u37+","u5-","u3-","u2-","u1+","u11-","u12+","u14+","u38-","u16+","u35+","u32-","u33+","u34+","u40-","u7-","u3-","u2-","u1+","u36-","u35-","u20-","u18-","u17+","u39+","u26-","u23-","u21-","u19-","u18-","u15-","u38+","u13-","u12-","u9-","u37+","u8+","u29+","u31+","u40-","u4-","u3+","u6+","u28+","u33-","u22-"]

filepath, filename, tangle_out_dir, graph, time_limit, Q, offset, T_max, V, solver = setup(*sys.argv)
nodes = list(graph.nodes)

import re
from math import floor


best_path_sample = np.zeros((T_max, V+1))
for i in range(len(best_path_nodes)):
    node_matches = re.search(
        r'(.+)([\+\-])',
        best_path_nodes[i]
    )
    node_name = f"{node_matches[1]}"
    best_path_sample[i, nodes.index(node_name)] = 1
best_path_sample[i+1:, -1] = 1
best_sample = best_path_sample.reshape((T_max * (V+1)),)

print(best_sample @ Q @ best_sample + offset)