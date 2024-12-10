#!/bin/bash

usage()
{
    echo "usage: build_qubo_input [[[-f file] [-m memory]] | [-h]]"
}

while [ "$1" != "" ]; do
    case $1 in
        -f | --file )           shift
                                filepath="$1"
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

if [ -f "$filepath" ]; then
    echo "Reading file: $filepath"
else
    echo "Could not find input file."
    exit 1
fi

filename=$(basename -- "$filepath")

case $memory in
    [0-9]* ) echo "Memory:" $memory
             ;;
    *      ) echo "Memory was not a number."; exit 1
esac

## MAIN
WORKING_DIR=/nfs/users/nfs_j/jc59/quantumwork/pangenome/qubo_solvers
source ~/.venv/qubo/bin/activate
outdir="$SCRATCH/out/tangle"

bsub -J  "build_qubo" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -M "$memory" -G "qpg" \
 -o "$outdir/build.$filename.%J" -e "$outdir/error.build.$filename.%J" -q qpg -gpu - \
 "python3 $WORKING_DIR/qubo_solvers/tangle/build_tangle_qubo_matrix.py $filepath"
exit 0 