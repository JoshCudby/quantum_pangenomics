#!/bin/bash

usage()
{
    echo "usage: compile_hubo.sh [[-f file -m memory -p reps -n shots -g num_gpu -i init -R rows -C cols -d swap_depth] | [-h]]"
}

memory="4000"
shots=4000

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
        -N | --nodes )          shift
                                nodes="$1"
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
outdir="$SCRATCH/new_qubo_formulation/oriented"

echo "QUBO Nonvar"
bsub -J "nonvar_qubo_$filename" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -M "$memory"\
 -o "$outdir/nonvariational.$filename.%J" -e "$outdir/error.nonvariational.$filename.%J" -G "qpg" -q "qpg" \
 "python3 $WORKING_DIR/nonvariational.py -f $filename -N $nodes -n $shots"

exit 0


