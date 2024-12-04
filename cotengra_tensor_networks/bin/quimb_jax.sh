#!/bin/bash

## MAIN

memory=128000
# outdir="/lustre/scratch127/qpg/jc59/out/cotengra"
outdir="out"

# Cotengra Testing
echo "Cotengra Testing"
bsub -R '"select[mem>'$memory'] rusage[mem='$memory']"' -M "$memory"\
 -o "$outdir/quimb_jax_real_data_greedy_nevergrad_equil_32.%J" -e "$outdir/error.quimb_jax_real_data_greedy_nevergrad_equil_32.%J"\
 -n 32\
 -G "qpg" -q "qpg" -gpu - "python3 quimb_jax.py"

exit 0