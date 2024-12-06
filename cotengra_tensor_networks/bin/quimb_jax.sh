#!/bin/bash

## MAIN

memory=128000
outdir="$SCRATCH/out/cotengra"
WORKING_DIR=/nfs/users/nfs_j/jc59/quantumwork/pangenome/cotengra_tensor_networks
source cotengra_venv/bin/activate

# Cotengra Testing
echo "Cotengra Testing"
bsub -R '"select[mem>'$memory'] rusage[mem='$memory']"' -M "$memory"\
 -o "$outdir/quimb_jax_real_data_greedy_nevergrad_equil_32.%J" -e "$outdir/error.quimb_jax_real_data_greedy_nevergrad_equil_32.%J"\
 -n 32\
 -G "qpg" -q "qpg" -gpu - "python3 quimb_jax.py"

exit 0