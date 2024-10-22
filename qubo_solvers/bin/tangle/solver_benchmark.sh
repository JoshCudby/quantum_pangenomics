#!/bin/bash

usage()
{
    echo "usage: solver_benchmark [[[-f file ] [-n normalisation] [-ct classical time limit] [-qt quantum time limit]] | [-h]]"
}

# Defaults
memory=4000
classical_time_limit=60
quantum_time_limit=0
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
                                  quantum_time_limit="$1"
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
    echo "Reading file:"; echo $filename
else
    echo "Could not find input file."
    exit 1
fi

case $normalisation in
    [0-9]* ) echo "Normalising node weights by:"; echo $normalisation
             ;;
    *      ) echo "Normalisation was not a number."; exit 1
esac

case $classical_time_limit in
    [0-9]* ) echo "Classical time limit:" $classical_time_limit
             ;;
    *      ) echo "Classical time limit was not a number."; exit 1
esac

# Gurobi solver
printf "\n\n"
echo "Gurobi Solver"
python3 "./tangle/max_path_gurobi.py" $filename $normalisation $time_limit

# MQLib solver
printf "\n\n"
echo "MQLib Solver"
python3 "./tangle/max_path_mqlib.py" $filename $normalisation $time_limit

# D-Wave solver
printf "\n\n"
echo "D-Wave Solver"
python3 "./tangle/max_path_dwave.py" $filename $normalisation $quantum_time_limit q

exit 0