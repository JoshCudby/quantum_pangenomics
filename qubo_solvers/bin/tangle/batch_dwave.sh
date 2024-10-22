#!/bin/bash

usage()
{
    echo "usage: batch_dwave [[[-f file ] [-n normalisation] [-qt quantum time limit] [-m memory] [-j jobs]] | [-h]]"
}

# Defaults
memory=4000
quantum_time_limit=-1
normalisation=1
jobs=1

while [ "$1" != "" ]; do
    case $1 in
        -f | --file )             shift
                                  filename="$1"
                                  ;;
        -n | --normalisation )    shift
                                  normalisation="$1"
                                  ;;
        -qt | --quantum-time )    shift
                                  quantum_time_limit="$1"
                                  ;;
        -m | --memory )           shift
                                  memory="$1"
                                  ;;
        -j | --jobs )             shift
                                  jobs="$1"
                                  ;;
        -h | --help )             usage
                                  exit
                                  ;;
        * )                       usage
                                  exit 1
    esac
    shift
done

if [ -f "./data/"$filename ]; then
    echo "Reading file:" $filename
else
    echo "Could not find input file."
    exit 1
fi

case $normalisation in
    [0-9]* ) echo "Normalising node weights by:" $normalisation
             ;;
    *      ) echo "Normalisation was not a number."; exit 1
esac

case $memory in
    [0-9]* ) echo "Memory:" $memory
             ;;
    *      ) echo "Memory was not a number."; exit 1
esac

case $jobs in
    [0-9]* ) echo "Jobs:" $jobs
             ;;
    *      ) echo "Jobs was not a number."; exit 1
esac

case $quantum_time_limit in
    -1     ) echo "Default quantum time limit"
             ;;
    [0-9]* ) echo "Quantum time limit:" $quantum_time_limit
             ;;
    *      ) echo "Quantum time limit was not a number."; exit 1
esac

## MAIN

# D-Wave solver
printf "\n\n"
echo "D-Wave Solver"
bsub -J  "dwaveJobs[1-$jobs]" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -M "$memory" -o "out/dwave.$filename.%J.%I" -e "out/error.dwave.$filename.%J" -G "qpg" "python3 ./tangle/max_path_dwave.py $filename $normalisation $quantum_time_limit q"

exit 0