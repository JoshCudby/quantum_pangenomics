#!/bin/bash


memory="128000"


## MAIN

WORKING_DIR="/nfs/users/nfs_j/jc59/quantumwork/pangenome/new_qubo_formulation/qubo_qaoa/nonvariational"
source "/nfs/users/nfs_j/jc59/quantumwork/pangenome/.venv/bin/activate"
outdir="$SCRATCH/new_qubo_formulation/oriented/circuit_depth_width"

echo "QUBO Circuit Depth"
bsub -J "qubo_depth_width" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -M "$memory"\
 -o "$outdir/qubo_depth_width.%J" -e "$outdir/error.qubo_depth_width.%J" -G "qpg" -q "qpg" \
 "python3 $WORKING_DIR/get_circuit_depth_widths.py"

exit 0


