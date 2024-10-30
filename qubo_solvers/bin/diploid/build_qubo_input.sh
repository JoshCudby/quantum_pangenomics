#!/bin/bash

usage()
{
    echo "usage: build_qubo_input [[[-f file] [-n normalisation] [-m memory]] | [-h]]"
}

while [ "$1" != "" ]; do
    case $1 in
        -f | --file )           shift
                                filepath="$1"
                                ;;
        -n | --normalisation )  shift
                                normalisation="$1"
                                ;;
        -m | --memory )         shift
                                memory="$1"
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done

if [ -f "$filepath" ]; then
    echo "Reading file: $filepath"
else
    echo "Could not find input file."
    exit 1
fi

filename=$(basename -- "$filepath")

case $normalisation in
    [0-9]* ) echo "Normalising node weights by: $normalisation"
             ;;
    *      ) echo "Normalisation was not a number."; exit 1
esac

case $memory in
    [0-9]* ) echo "Memory: $memory"
             ;;
    *      ) echo "Memory was not a number."; exit 1
esac


## MAIN
WORKING_DIR=/nfs/users/nfs_j/jc59/quantumwork/pangenome/qubo_solvers
source ~/pangenome/bin/activate

bsub -J  "build_qubo" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -M "$memory" -G "qpg" \
-o "./out/diploid/build.$filename.%J" -e "./out/diploid/error.build.$filename.%J" \
"python3 ./diploid_tangle/build_diploid_qubo_matrix.py $filepath $normalisation"