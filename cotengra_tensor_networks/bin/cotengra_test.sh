#!/bin/bash

## MAIN

memory=64000

WORKING_DIR=/nfs/users/nfs_j/jc59/quantumwork/pangenome/qokit_simulation
source ~/cotengra-venv/bin/activate

# Cotengra Testing
echo "Cotengra Testing"
bsub -R '"select[mem>'$memory'] rusage[mem='$memory']"' -M "$memory" -o "out/cotengra.test.%J" -e "out/error.cotengra.test.%J" -G "qpg" -q "normal" "python3 ./cotengra_testing.py"

exit 0