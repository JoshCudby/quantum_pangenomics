#!/bin/bash
usage()
{
    echo "usage: run_small_nonvariational.sh [[-n shots] | [-h]]"
}

# qubits (4 8 9 12 12)
filenames=("test_N2_W2" "test_N7_W2" "trivial"  'test_N7_W3' "test_N3_W4")
copy_numbers=("1,1" "1,0,0,0,0,0,1" "1,1,1" "1,1,0,0,0,0,1" "2,1,1")
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
    c="${copy_numbers[i]}"
    nonvariational.sh -f "$filename" -m 8000 -n "$shots" -c "$c"
done
