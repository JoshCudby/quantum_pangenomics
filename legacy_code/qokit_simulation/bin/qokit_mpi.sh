#!/bin/bash
## MAIN

memory=32000

module load cuda-12.1.1
module load /software/modules/ISG/experimental/fg12/openmpi/4.0.3-cuda
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
WORKING_DIR=/nfs/users/nfs_j/jc59/quantumwork/pangenome/qokit_simulation
outdir="$SCRATCH/out/qokit"
# export OMPI_COMM_WORLD_LOCAL_RANK
# export OMPI_COMM_WORLD_SIZE

# QOKit Testing
echo "QOKit MPI Testing"
bsub -R '"select[mem>'$memory'] rusage[mem='$memory'] span[ptile=64]"' -n 4 -M "$memory"\
 -gpu "num=4:gmem=80000:mode=shared:block=yes"\
 -o "$outdir/qokit_mpi.test.%J" -e "$outdir/error.qokit_mpi.test.%J" -G "qpg" -q "qpg" \
 "mpiexec python $WORKING_DIR/qokit_mpi.py"

exit 0