#!/bin/bash

usage()
{
    echo "usage: param_exploration.sh [[-f file -m memory -c copy_numbers] | [-h]]"
}

memory="4000"

while [ "$1" != "" ]; do
    case $1 in
        -f | --file )           shift
                                filename="$1"
                                ;;
        -m | --memory )         shift
                                memory="$1"
                                ;;
        -c | --copy-numbers )   shift
                                copy_numbers="$1"
                                ;;
        --normalise )           normalise="--normalise"
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

WORKING_DIR="/nfs/users/nfs_j/jc59/quantumwork/pangenome/new_hubo_formulation/hubo_qaoa/nonvariational"
source "/nfs/users/nfs_j/jc59/quantumwork/pangenome/.venv/bin/activate"
outdir="$SCRATCH/new_hubo_formulation/nonvariational/param_exploration"

echo "HUBO Nonvar"
bsub -J "hubo_param_$filename" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -M "$memory" -gpu - \
 -o "$outdir/param.$filename.%J" -e "$outdir/error.param.$filename.%J" -G "qpg" -q "qpg" \
 "python3 $WORKING_DIR/param_exploration.py -f $filename -c $copy_numbers $normalise"

exit 0


# dict_keys(['test_N2_W2', 'trivial', 'test_N3_W4', 'test_N4_W5', 'test_N7_W2', 'test_N7_W3', 'test_N7_W4', 'test_N8_W2', 'test_N8_W3'])