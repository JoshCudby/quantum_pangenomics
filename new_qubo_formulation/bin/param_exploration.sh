#!/bin/bash

usage()
{
    echo "usage: param_exploration.sh [[-f file -m memory ] | [-h]]"
}

memory="4000"

while [ "$1" != "" ]; do
    case $1 in
        -f | --file )           shift
                                filename="$1"
                                ;;
        -N | --nodes )          shift
                                nodes="$1"
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

## MAIN

WORKING_DIR="/nfs/users/nfs_j/jc59/quantumwork/pangenome/new_qubo_formulation/qubo_qaoa/nonvariational"
source "/nfs/users/nfs_j/jc59/quantumwork/pangenome/.venv/bin/activate"
outdir="$SCRATCH/new_qubo_formulation/oriented/param_exploration"

echo "QUBO Nonvar"
bsub -J "LR_param_exploration.$filename" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -M "$memory" -gpu - \
 -o "$outdir/param_exploration.$filename.%J" -e "$outdir/error.param_exploration.$filename.%J" -G "qpg" -q "qpg" \
 "python3 $WORKING_DIR/param_exploration.py -f $filename -N $nodes"

exit 0


