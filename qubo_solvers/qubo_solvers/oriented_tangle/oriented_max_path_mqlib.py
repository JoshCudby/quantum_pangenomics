import numpy as np
import sys
import subprocess
import os
from datetime import datetime
from qubo_solvers.definitions import MQLIB_DIR
from qubo_solvers.oriented_tangle.utils.setup_utils import setup
from qubo_solvers.oriented_tangle.utils.graph_utils import oriented_graph_from_file, normalise_node_weights
from qubo_solvers.oriented_tangle.utils.sampling_utils import validate_path, sample_list_to_path

filepath, filename, oriented_out_dir, graph, time_limit, Q, offset, T_max, V = setup(sys.argv)
           
mqlib_input_path = f'{oriented_out_dir}/mqlib_input_{filename}.txt'
# Run the MQLib solver and capture output
process = subprocess.run([f"{MQLIB_DIR}/bin/MQLib", "-fQ", filepath, "-h", "PALUBECKIS2004bMST2", "-r", str(time_limit), "-ps"], capture_output=True)
out = process.stdout.decode("utf-8")

# First line of output includes run data. 3rd line contains the solution.
out_data = [x for x in out.split('\n') if len(x) > 0]
solution = out_data[2].split()
solution = np.array([int(x) for x in solution])
solution_energy = int(out_data[0].split(',')[3])
energy = offset - solution_energy
path = sample_list_to_path(solution, graph, T_max, V)

validate_path(path, graph)
print(f"Energy of path: {energy}")

now = datetime.now().strftime("%d%m%Y_%H%M")
save_file = oriented_out_dir + f"/mqlib_{filename}_{now}"   
    
to_save = np.array([solution, energy, path], dtype=object)
np.save(save_file, to_save)
print('Compilation Data')
print(f'[{time_limit}, {energy}],')