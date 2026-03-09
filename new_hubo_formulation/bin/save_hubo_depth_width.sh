#!/bin/bash

usage()
{
    echo "usage: param_exploration.sh [[-f file -m memory -c copy_numbers] | [-h]]"
}

memory="256000"

## MAIN

WORKING_DIR="/nfs/users/nfs_j/jc59/quantumwork/pangenome/new_hubo_formulation/hubo_qaoa/nonvariational/plotting"
source "/nfs/users/nfs_j/jc59/quantumwork/pangenome/.venv/bin/activate"
outdir="$SCRATCH/new_hubo_formulation/circuit_depths"

bsub -J "hubo_plot" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -M "$memory" -gpu - \
 -o "$outdir/plot_hubo_depth.%J" -e "$outdir/error.plot_hubo_depth.%J" -G "qpg" -q "qpg" \
 "python3 $WORKING_DIR/save_circuit_depth_width.py"

exit 0


# dict_keys(['test_N2_W2', 'trivial', 'test_N3_W4', 'test_N4_W5', 'test_N7_W2', 'test_N7_W3', 'test_N7_W4', 'test_N8_W2', 'test_N8_W3'])