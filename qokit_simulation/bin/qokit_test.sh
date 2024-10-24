#!/bin/bash

## MAIN

memory=2000

module load cuda-12.1.1
WORKING_DIR=/nfs/users/nfs_j/jc59/quantumwork/pangenome/qokit_simulation
source ~/qokit_311/bin/activate

# QOKit Testing
echo "QOKit Testing"
bsub -R '"select[mem>'$memory'] rusage[mem='$memory']"' -gpu - -M "$memory" -o "out/qokit.test.%J" -e "out/error.qokit.test.%J" -G "qpg" -q "qpg" "python3 ./qokit_job_testing.py"

exit 0