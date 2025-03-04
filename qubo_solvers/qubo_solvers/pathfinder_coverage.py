import subprocess
import os
from qubo_solvers.definitions import QUANTUM_DIR

def run_pathfinder_coverage(out_dir, gfa_file, coverage_suffix):
    filename = os.path.basename(gfa_file)
    # do_adjust = filename not in ['trivial.gfa', 'small_test.gfa', 'test.gfa']
    do_adjust = False
    to_save = f'{out_dir}/{filename}.{coverage_suffix}'
    
    pathfinder = "jkbpathfinder"
    print(f"Using pathfinder from: {pathfinder}")
    args = [f"{QUANTUM_DIR}/{pathfinder}/pathfinder", "-o", to_save, "-C40"]
    if do_adjust:
        args.append("-a")
    args.append(gfa_file)
    subprocess.run(args, capture_output=False)
    
    with open(to_save, 'r') as f:
        lines = f.readlines()
    if not len(lines):
        raise Exception(f'Could not read copy numbers from {to_save}')
    
    # TODO: process subgraphs?
    nodes = lines[0].split()
    copy_numbers = [int(x) for x in lines[1].split()]
    return copy_numbers, nodes