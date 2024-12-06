#!/bin/bash

## MAIN
export COTENGRA_NUM_WORKERS=32
memory=128000
outdir="$SCRATCH/out/cotengra"
WORKING_DIR=/nfs/users/nfs_j/jc59/quantumwork/pangenome/cotengra_tensor_networks
source cotengra_venv/bin/activate

# Cotengra Testing
echo "Cotengra Testing"
bsub -R '"select[mem>'$memory'] rusage[mem='$memory']"' -M "$memory"\
 -o "$outdir/trivial_32core_1_gpu.%J" -e "$outdir/error.trivial_32core_1_gpu.%J"\
 -n 32\
 -G "qpg" -q "qpg" -gpu - "python3 quimb_jax.py"

exit 0