#!/bin/bash

## MAIN

memory=32000

module load cuda-12.1.1
module load ISG/experimental/fg12/openmpi/5.0.4-cuda12.1-lsf
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
WORKING_DIR=/nfs/users/nfs_j/jc59/quantumwork/pangenome/qokit_simulation
outdir="$SCRATCH/out/qokit"
# export OMPI_COMM_WORLD_LOCAL_RANK
# export OMPI_COMM_WORLD_SIZE

# QOKit Testing
echo "QOKit MPI Testing"
bsub -R '"select[mem>'$memory'] rusage[mem='$memory']"' -n 32 -gpu "num=2" -M "$memory"\
 -o "$outdir/qokit_mpi.test.%J" -e "$outdir/error.qokit_mpi.test.%J" -G "qpg" -q "qpg" \
 "mpiexec python $WORKING_DIR/qokit_mpi.py"

exit 0