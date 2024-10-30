#!/bin/bash

usage()
{
    echo "usage: build_qubo_input [[[-f file] [-n normalisation] [-m memory]] | [-h]]"
}

while [ "$1" != "" ]; do
    case $1 in
        -f | --file )           shift
                                filename="$1"
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

if [[ $filename =~ ^data/(.*)$ ]]; then
    filename="${BASH_REMATCH[1]}" 
fi

if [ -f "$filename" ]; then
    echo "Reading file: $filename"
else
    echo "Could not find input file."
    exit 1
fi

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
bsub -J  "build_qubo" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -M "$memory" -G "qpg" \
-o "build.$filename.%J" -e "error.build.$filename.%J" \
"python3 ./diploid_tangle/build_diploid_qubo_matrix.py $filename $normalisation"
