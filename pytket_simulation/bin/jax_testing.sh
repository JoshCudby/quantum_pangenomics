#!/bin/bash

module load cuda-12.1.1
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64

memory=4000
num_gpu=3
WORKING_DIR=/nfs/users/nfs_j/jc59/quantumwork/pangenome/pytket_simulation/pytket_qaoa
outdir="$SCRATCH/out/pytket"
mkdir -p $outdir

# Jax Testing
echo "Jax Testing"
bsub -J "jax" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -gpu "num=$num_gpu:aff=no:j_exclusive=yes" -M "$memory"\
 -o "$outdir/jax.%J" -e "$outdir/error.jax.test.%J" -G "qpg" -q "qpg" \
 "python3 $WORKING_DIR/jax_test.py"

exit 0