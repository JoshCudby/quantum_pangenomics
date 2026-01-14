#!/bin/bash
usage()
{
    echo "usage: run_small_qubo_nonvariational.sh [[-n shots] | [-h]]"
}

filenames=("test_N2_W2" "trivial" "test_N3_W4" "test_N7_W2" "test_N4_W5")
Ns=(2 3 3 7 4)
shots=4000

while [ "$1" != "" ]; do
    case $1 in
        -n | --shots )          shift
                                shots="$1"
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done

for i in "${!filenames[@]}"; do
    filename="${filenames[i]}"
    N="${Ns[i]}"
    qubo_nonvariational.sh -f "$filename" -m 4000 -n "$shots" -N "$N"
done
