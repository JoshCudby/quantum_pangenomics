#!/bin/bash

usage()
{
    echo "usage: solver_benchmark_bsub [[[-f file ] [-n normalisation] [-ct classical time limit] [-qt quantum time limit] [-m memory]] | [-h]]"
}

# Defaults
memory=4000
classical_time_limit=60
quantum_time_limit=-1
normalisation=1

while [ "$1" != "" ]; do
    case $1 in
        -f | --file )             shift
                                  filename="$1"
                                  ;;
        -n | --normalisation )    shift
                                  normalisation="$1"
                                  ;;
        -ct | --classical-time )  shift
                                  classical_time_limit="$1"
                                  ;;
        -qt | --quantum-time )    shift
                                  qunatum_time_limit="$1"
                                  ;;
        -m | --memory )           shift
                                  memory="$1"
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

case $classical_time_limit in
    [0-9]* ) echo "Classical time limit:" $classical_time_limit
             ;;
    *      ) echo "Classical time limit was not a number."; exit 1
esac

case $memory in
    [0-9]* ) echo "Memory:" $memory
             ;;
    *      ) echo "Memory was not a number."; exit 1
esac

## MAIN

# Gurobi solver
printf "\n\n"
echo "Gurobi Solver"
bsub -R '"select[mem>'$memory'] rusage[mem='$memory']"' -M "$memory" -o "out/gurobi.$filename.%J" -e "out/error.gurobi.$filename.%J" -G "qpg" "python3 ./tangle/max_path_gurobi.py $filename $normalisation $classical_time_limit"

# MQLib solver
printf "\n\n"
echo "MQLib Solver"
bsub -R '"select[mem>'$memory'] rusage[mem='$memory']"' -M "$memory" -o "out/mqlib.$filename.%J" -e "out/error.mqlib.$filename.%J" -G "qpg" "python3 ./tangle/max_path_mqlib.py $filename $normalisation $classical_time_limit"

# D-Wave solver
printf "\n\n"
echo "D-Wave Solver"
bsub -R '"select[mem>'$memory'] rusage[mem='$memory']"' -M "$memory" -o "out/dwave.$filename.%J" -e "out/error.dwave.$filename.%J" -G "qpg" "python3 ./tangle/max_path_dwave.py $filename $normalisation $quantum_time_limit q"

exit 0