#!/bin/bash

## MAIN

memory=32000


# Cotengra Testing
echo "Cotengra Testing"
bsub -R '"select[mem>'$memory'] rusage[mem='$memory']"' -M "$memory"\
 -o "out/cotengra.test_2.%J" -e "out/error.cotengra.test_2.%J"\
 -n 32\
 -G "qpg" -q "qpg" -gpu - "python3 cotengra_testing_2.py"

exit 0