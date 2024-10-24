#!/bin/bash

## MAIN

memory=2000

mkdir -p out/qokit
# QOKit Testing
echo "QOKit Testing"
bsub -R '"select[mem>'$memory'] rusage[mem='$memory']"' -gpu - -M "$memory" -o "out/qokit.test.%J" -e "out/error.qokit.test.%J" -G "qpg" -q "qpg" "python3 ./qokit_job_testing.py"

exit 0