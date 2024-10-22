#!/bin/bash

usage()
{
    echo "usage: batch_mqlib [[[-f file ] [-n normalisation] [-t time limit] [-m memory] [-j jobs]] | [-h]]"
}

# Defaults
memory=4000
time_limit=-1
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
        -t | --time )             shift
                                  time_limit="$1"
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

case $time_limit in
    -1     ) echo "Default time limit"
             ;;
    [0-9]* ) echo "Time limit:" $time_limit
             ;;
    *      ) echo "Time limit was not a number."; exit 1
esac

## MAIN

# MQLib solver
printf "\n\n"
echo "MQlib Solver"
bsub -J  "mqlibJobs[1-$jobs]" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -M "$memory" -o "out/mqlib.$filename.%J.%I" -e "out/error.mqlib.$filename.%J" -G "qpg" "python3 ./tangle/max_path_mqlib.py $filename $normalisation $time_limit"

exit 0