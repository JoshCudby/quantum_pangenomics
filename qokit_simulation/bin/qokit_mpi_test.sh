#!/bin/bash

## MAIN

memory=4000

module load cuda-12.1.1
LD_LIBRARY_PATH=$CUDA_HOME/lib64
WORKING_DIR=/nfs/users/nfs_j/jc59/quantumwork/pangenome/qokit_simulation
source ~/qokit-311/bin/activate

# QOKit Testing
echo "QOKit MPI Testing"
bsub -R '"select[mem>'$memory'] rusage[mem='$memory']"' -gpu "num=2" -M "$memory" -o "out/qokit.mpi.test.%J" -e "out/error.qokit.mpi.test.%J" -G "qpg" -q "qpg" "python3 ./qokit_mpi_testing.py"

exit 0