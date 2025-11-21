#!/bin/bash

usage()
{
    echo "usage: compile_hubo.sh [[-f file -m memory -p reps -n shots -g num_gpu -i init -R rows -C cols -d swap_depth] | [-h]]"
}

memory="4000"
timeout=60

while [ "$1" != "" ]; do
    case $1 in
        -f | --file )           shift
                                filename="$1"
                                ;;
        -m | --memory )         shift
                                memory="$1"
                                ;;
        -t | --timeout )        shift
                                timeout="$1"
                                ;;
        -C | --coupling )       shift
                                coupling="$1"
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

WORKING_DIR="/nfs/users/nfs_j/jc59/quantumwork/pangenome/new_hubo_formulation/hubo_qaoa"
source "/nfs/users/nfs_j/jc59/quantumwork/pangenome/.venv/bin/activate"
outdir="$SCRATCH/new_hubo_formulation/circuit_depths"

echo "HUBO Testing"
bsub -J "hubo_depths" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -gpu "num=1:aff=no:j_exclusive=yes" -M "$memory"\
 -o "$outdir/depths.$timeout.%J" -e "$outdir/error.depths.$timeout.%J" -G "qpg" -q "qpg" \
 "python3 $WORKING_DIR/circuit_depths.py -t $timeout -C $coupling"

exit 0


