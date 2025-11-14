#!/bin/bash

usage()
{
    echo "usage: compile_hubo.sh [-t timeout] | [-h]]"
}

timeout=60
memory=64000

while [ "$1" != "" ]; do
    case $1 in
        -t | --timeout )        shift
                                timeout="$1"
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

WORKING_DIR="/nfs/users/nfs_j/jc59/quantumwork/pangenome/qiskit_simulation/qiskit_qaoa/hubo"
source "/nfs/users/nfs_j/jc59/quantumwork/pangenome/.venv/bin/activate"
outdir="$SCRATCH/circuit_depths"

echo "Circuit depths"
bsub -J "circuit_depths" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -gpu "num=1:aff=no:j_exclusive=no" -M "$memory"\
 -o "$outdir/depths_precompute.%J" -e "$outdir/error.depths_precompute.%J" -G "qpg" -q "qpg" \
 "python3 $WORKING_DIR/get_circuit_depths_precompute.py -t $timeout"

exit 0


