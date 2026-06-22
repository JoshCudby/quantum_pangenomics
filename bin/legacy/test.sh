#!/bin/bash

QUBO_DIR=/nfs/users/nfs_j/jc59/quantumwork/pangenome/qubo_solvers
source $QUBO_DIR/.venv/bin/activate

gfa_filepath=/lustre/scratch127/qpg/jc59/full_benchmark/oriented/301.24022025.1129/1/assembled.syncasm.utg.final.gfa
outdir=/lustre/scratch127/qpg/jc59/full_benchmark/oriented/301.24022025.1129/1
num_jobs=2
time_limits="3,5"

python3 "$QUBO_DIR"/qubo_solvers/oriented_tangle/oriented_max_path.py -s "mqlib" -f "$gfa_filepath" -d "$outdir" -j "$num_jobs" -t $time_limits