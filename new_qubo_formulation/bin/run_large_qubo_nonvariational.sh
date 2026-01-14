#!/bin/bash
usage()
{
    echo "usage: run_large_qubo_nonvariational.sh [[-n shots] | [-h]]"
}

filenames=("test_N4_W6" "test_N5_W6" "test_N8_W4" "test_N7_W5" "test_N8_W5")
Ns=(4 5 8 7 8)
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
    qubo_nonvariational.sh -f "$filename" -m 8000 -n "$shots" -N "$N"
done
