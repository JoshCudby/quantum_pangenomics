#!/bin/bash
usage()
{
    echo "usage: run_large_nonvariational.sh [[-n shots] | [-h]]"
}

# qubits (15 16 20 24)
filenames=('test_N4_W5' 'test_N7_W4' 'test_N8_W5' 'test_N8_W6')
copy_numbers=("2,1,1,1" "1,1,1,0,0,0,1" "1,1,1,1,0,0,0,1" "1,1,0,1,1,1,0,1")
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
    nonvariational.sh -f "$filename" -m 16000 -n "$shots" -c "$c"
done
