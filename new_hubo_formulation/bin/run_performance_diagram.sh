#!/bin/bash
usage()
{
    echo "usage: run_small_performance_diagram.sh [[-n shots] | [-h]]"
}

# qubits (4 8 9 12 12)
filenames=("test_N2_W2" "test_N7_W2" "trivial" 'test_N3_W4' "test_N4_W5")
copy_numbers=("1,1" "1,0,0,0,0,0,1" "1,1,1" "2,1,1" "2,1,1,1")


for i in "${!filenames[@]}"; do
    filename="${filenames[i]}"
    c="${copy_numbers[i]}"
    performance_diagram.sh -f "$filename" -m 16000 -c "$c"
done
