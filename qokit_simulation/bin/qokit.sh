#!/bin/bash

## MAIN

memory=32000

module load cuda-12.1.1
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
WORKING_DIR=/nfs/users/nfs_j/jc59/quantumwork/pangenome/qokit_simulation
outdir="$SCRATCH/out/qokit"

# QOKit Testing
echo "QOKit Testing"
bsub -R '"select[mem>'$memory'] rusage[mem='$memory']"' -n 32 -gpu "num=2" -M "$memory"\
 -o "$outdir/qokit.test.%J" -e "$outdir/error.qokit.test.%J" -G "qpg" -q "qpg" \
 "python3 $WORKING_DIR/qokit.py"

exit 0