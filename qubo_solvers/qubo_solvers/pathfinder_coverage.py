from pathfinder import pathfinder
import os

def run_pathfinder_coverage(out_dir, gfa_file, coverage_suffix):
    filename = os.path.basename(gfa_file)
    max_copy = 10
    min_cfrac = 0
    max_path = 1000000
    do_part = False
    do_adjust = filename not in ['trivial.gfa', 'small_test.gfa', 'test.gfa']
    s_source = None
    s_target = None
    ec_tag = None
    kc_tag = None
    sc_tag = "SC:f"
    VERBOSE = 0

    to_save = f'{out_dir}/{filename}.{coverage_suffix}'

    pathfinder(
        gfa_file, max_copy, min_cfrac, max_path, do_part, do_adjust, s_source, s_target, 
        ec_tag, kc_tag, sc_tag, VERBOSE, to_save
    )
    
    with open(to_save, 'r') as f:
        lines = f.readlines()
    if len(lines) < 3:
        raise Exception(f'Could not read copy numbers from {to_save}')
    copy_numbers = [int(x) for x in lines[2].split()]
    return copy_numbers