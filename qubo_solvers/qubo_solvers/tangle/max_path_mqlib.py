import numpy as np
import sys
import os
import subprocess
from datetime import datetime
from qubo_solvers.definitions import DATA_DIR, MQLIB_DIR, OUT_DIR
from qubo_solvers.tangle.utils.sampling_utils import qubo_vars_to_path
from qubo_solvers.tangle.utils.graph_utils import graph_from_gfa_file, normalise_node_weights
from qubo_solvers.tangle.utils.sampling_utils import validate_path


if len(sys.argv) > 1:
    filepath = sys.argv[1]
else:
    filepath = f"{DATA_DIR}/test.gfa"

if len(sys.argv) > 2:
    try:
        normalisation = int(sys.argv[2])
    except ValueError:
        normalisation = 1
else:
    normalisation = 1

if len(sys.argv) > 3:
    try:
        time_limit = int(sys.argv[3])
    except ValueError:
        time_limit = 3
else:
    time_limit = 3
    
filename = os.path.basename(filepath)
    
tangle_out_dir = f"{OUT_DIR}/tangle"
input_filepath = f"{tangle_out_dir}/mqlib_input_{filename}.txt"
qubo_data_filepath = f"{tangle_out_dir}/qubo_data_{filename}.npy"

Q, offset, T_max, V = np.load(qubo_data_filepath, allow_pickle=True)

graph = graph_from_gfa_file(f"{DATA_DIR}/{filename}")
print(f"Normalising by: {normalisation}")
graph = normalise_node_weights(graph, normalisation)


# Run the MQLib solver and capture output
process = subprocess.run([f"{MQLIB_DIR}/bin/MQLib", "-fQ", input_filepath, "-h", "PALUBECKIS2004bMST2", "-r", str(time_limit), "-ps"], capture_output=True)
out = process.stdout.decode("utf-8")

# First line of output includes run data. 3rd line contains the solution.
out_data = [x for x in out.split('\n') if len(x) > 0]
solution = out_data[2].split()
solution = [int(x) for x in solution]
solution_energy = int(out_data[0].split(',')[3])
energy = offset - solution_energy

path = qubo_vars_to_path(solution, graph)
validate_path(path, graph)
print(f"Energy of path: {energy}")
    
now = datetime.now().strftime("%d%m%Y_%H%M")
save_file = tangle_out_dir + f"/mqlib_{filename}_{now}"   
    
to_save = np.array([solution, energy, path], dtype=object)
np.save(save_file, to_save)
print('Compilation Data')
print(f'[{time_limit}, {energy}],')

