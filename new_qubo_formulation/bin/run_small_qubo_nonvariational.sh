#!/bin/bash

filenames=("test_N2_W2" "trivial" "test_N3_W4" "test_N7_W2" "test_N4_W5")
Ns=(2 3 3 7 4)

for i in "${!filenames[@]}"; do
    filename="${filenames[i]}"
    N="${Ns[i]}"
    qubo_nonvariational.sh -f "$filename" -m 4000 -n 4000 -N "$N"
done
