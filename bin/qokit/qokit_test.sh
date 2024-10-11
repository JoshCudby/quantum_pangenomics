#!/bin/bash

## MAIN

memory=8000

mkdir -p out/qokit
# QOKit Testing
echo "QOKit Testing"
bsub -R '"select[mem>'$memory'] rusage[mem='$memory']"' -M "$memory" -o "out/qokit.test.%J" -e "out/error.qokit.test.%J" -G "qpg" "python3 ./qokit_simulation/qokit_job_testing.py"

exit 0