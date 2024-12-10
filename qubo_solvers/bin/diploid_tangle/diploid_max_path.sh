#!/bin/bash

usage()
{
    echo "usage: oriented_max_path [[[-f file] [-j jobs] [-t times] [-n normalisation] [-m memory] [-s solver]] | [-h]]"
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
        -j | --jobs )           shift
                                jobs="$1"
                                ;;
        -s | --solver )         shift
                                solver="$1"
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

if [ -f "$filepath" ]; then
    echo "Reading file:" $filename
else
    echo "Could not find input file."
    exit 1
fi
filename=$(basename -- "$filepath")

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

outdir="$SCRATCH/out/diploid"
mkdir -p $outdir

## MAIN
for time_limit in "${times_arr[@]}"
do
    echo Submitting $solver batch with time limit: $time_limit
    bsub -J  "dip_maxpath[1-$jobs]" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -q qpg -gpu - \
     -M "$memory" -o "$outdir/$solver.$filename.%J.%I" -e "$outdir/error.$solver.$filename.%J"\
     -G "qpg" "python3 qubo_solvers/diploid_tangle/diploid_max_path.py $solver $filepath $normalisation $time_limit"
done
exit 0