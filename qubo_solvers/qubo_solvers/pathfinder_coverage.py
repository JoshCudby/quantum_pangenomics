from pathfinder import pathfinder
import os

def run_pathfinder_coverage(gfa_file, coverage_suffix):
    max_copy = 10
    min_cfrac = 0
    max_path = 1000000
    do_part = False
    do_adjust = not os.path.basename(gfa_file) in ['trivial.gfa', 'test.gfa']
    s_source = None
    s_target = None
    ec_tag = None
    kc_tag = None
    sc_tag = None # "SC:f"
    VERBOSE = 0

    path = pathfinder(gfa_file, max_copy, min_cfrac, max_path, do_part, do_adjust, s_source, s_target, ec_tag, kc_tag, sc_tag, VERBOSE, f"{gfa_file}_{coverage_suffix}")
    return(path)