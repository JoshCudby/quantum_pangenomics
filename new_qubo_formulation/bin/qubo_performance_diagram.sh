#!/bin/bash

usage()
{
    echo "usage: param_exploration.sh [[-f file -m memory ] | [-h]]"
}

memory="4000"
shots="1"
measure=""
gpu_str="num=1"

while [ "$1" != "" ]; do
    case $1 in
        -f | --file )           shift
                                filename="$1"
                                ;;
        -m | --memory )         shift
                                memory="$1"
                                ;;
        -n | --shots )          shift
                                shots="$1"
                                ;;
        --measure )             measure="--measure"
                                gpu_str="num=0"
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done

## MAIN

WORKING_DIR="/nfs/users/nfs_j/jc59/quantumwork/pangenome/new_qubo_formulation/qubo_qaoa/nonvariational"
source "/nfs/users/nfs_j/jc59/quantumwork/pangenome/.venv/bin/activate"
outdir="$SCRATCH/new_qubo_formulation/oriented/param_exploration"

echo "QUBO Nonvar"
bsub -J "qubo_perf.$filename" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -M "$memory" \
 -o "$outdir/qubo_perf.$filename.%J" -e "$outdir/error.qubo_perf.$filename.%J" -gpu "$gpu_str" -G "qpg" -q "qpg"  \
 "python3 $WORKING_DIR/performance_diagram.py -f $filename -n $shots $measure"

exit 0


