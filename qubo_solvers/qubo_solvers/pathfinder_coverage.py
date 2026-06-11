"""Wrapper for the Pathfinder binary that extracts per-node copy numbers from a GFA file."""

import subprocess
import os
from qubo_solvers.definitions import QUANTUM_DIR

def run_pathfinder_coverage(out_dir, gfa_file, coverage_suffix):
    """Run the Pathfinder binary to compute per-node copy numbers for a GFA graph.

    Pathfinder performs exhaustive path enumeration over the sequence graph and
    writes the resulting node copy numbers to a two-line text file: the first
    line contains whitespace-separated node names, the second contains the
    corresponding integer copy numbers.

    Args:
        out_dir (str): Directory in which the output file will be written.
        gfa_file (str): Path to the input ``.gfa`` file.
        coverage_suffix (str): String appended to the GFA filename to form the
            output filename (e.g. ``"coverage"`` produces
            ``<out_dir>/<gfa_basename>.coverage``).

    Returns:
        tuple[list[int], list[str]]: A pair ``(copy_numbers, nodes)`` where
            ``copy_numbers`` is a list of integer copy numbers and ``nodes`` is
            the corresponding list of node name strings, both in the order
            returned by Pathfinder.

    Raises:
        Exception: If the output file is empty (Pathfinder produced no output).
    """
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