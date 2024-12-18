#!/bin/bash

## MAIN
export COTENGRA_NUM_WORKERS=32
memory=64000
outdir="$SCRATCH/out/cotengra"
WORKING_DIR=/nfs/users/nfs_j/jc59/quantumwork/pangenome/cotengra_tensor_networks
source $WORKING_DIR/cotengra_venv/bin/activate

# Cotengra Testing
echo "Cotengra Testing"
bsub -R '"select[mem>'$memory'] rusage[mem='$memory']"' -M "$memory"\
 -o "$outdir/trivial_non_local_exp.%J" -e "$outdir/error.trivial_non_local_exp.%J"\
 -n 32\
 -G "qpg" -q "qpg" -gpu - "python3 $WORKING_DIR/non_local_exp.py"

exit 0