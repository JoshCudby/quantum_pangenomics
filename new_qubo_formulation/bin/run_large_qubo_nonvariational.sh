#!/bin/bash

filenames=("test_N4_W6" "test_N5_W6" "test_N8_W4" "test_N7_W5" "test_N8_W5")
Ns=(4 5 8 7 8)

for i in "${!filenames[@]}"; do
    filename="${filenames[i]}"
    N="${Ns[i]}"
    qubo_nonvariational.sh -f "$filename" -m 8000 -n 4000 -N "$N"
done
