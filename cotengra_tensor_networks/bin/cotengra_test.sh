#!/bin/bash

## MAIN

memory=128000


# Cotengra Testing
echo "Cotengra Testing"
bsub -R '"select[mem>'$memory'] rusage[mem='$memory']"' -M "$memory"\
 -o "out/cotengra.test.%J" -e "out/error.cotengra.test.%J"\
 -n 80\
 -G "qpg" -q "qpg" -gpu - "python3 cotengra_testing.py"

exit 0