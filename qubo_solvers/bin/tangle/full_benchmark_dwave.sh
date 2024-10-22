#!/bin/bash

usage()
{
    echo "usage: full_benchmark_dwave [[[-f file] [-j jobs] [-t times] [-n normalisation] [-m memory]] | [-h]]"
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
        -j | --jobs )           shift
                                jobs="$1"
                                ;;
        -t | --times )          shift
                                times_arr=($1)
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
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

## MAIN
for time_limit in "${times_arr[@]}"
do
    echo Submitting batch with time limit: $time_limit
    bsub -J  "dwaveJobs[1-$jobs]" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -M "$memory" -o "out/dwave.full.$filename.%J.%I" -e "out/error.dwave.full.$filename.%J" -G "qpg" "python3 ./tangle/max_path_dwave.py $filename $normalisation $time_limit q"
done
