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
        -o | --out )            shift
                                outpath="$1"
                                ;;
        -c | --copy-numbers )   shift
                                copy_numbers="$1"
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
source $WORKING_DIR/qubo_venv/bin/activate
outdir="$SCRATCH/out/oriented"
mkdir -p $outdir

bsub -J  "build_qubo" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -M "$memory" -G "qpg" \
 -o "$outdir/build.$filename.%J" -e "$outdir/error.build.$filename.%J" -q qpg -gpu - \
 "python3 $WORKING_DIR/qubo_solvers/oriented_tangle/build_oriented_qubo_matrix.py -f $filepath -d $outpath" -p "10,5,1" -c $copy_numbers
exit 0