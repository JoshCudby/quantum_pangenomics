#!/bin/bash

usage()
{
    echo "usage: circuit_depths_new.sh [[-m memory] | [-h]]"
}

memory="4000"

while [ "$1" != "" ]; do
    case $1 in
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

WORKING_DIR="/nfs/users/nfs_j/jc59/quantumwork/pangenome/new_hubo_formulation/hubo_qaoa"
source "/nfs/users/nfs_j/jc59/quantumwork/pangenome/.venv/bin/activate"
outdir="$SCRATCH/new_hubo_formulation/circuit_depths"

echo "HUBO Testing"
bsub -J "hubo_depths" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -M "$memory"\
 -o "$outdir/depths.new.%J" -e "$outdir/error.depths.new.%J" -G "qpg" -q "qpg" \
 "python3 $WORKING_DIR/get_circuit_depths_all_to_all_new.py"

exit 0


