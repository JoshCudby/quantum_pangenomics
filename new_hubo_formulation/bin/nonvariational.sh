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

## MAIN

WORKING_DIR="/nfs/users/nfs_j/jc59/quantumwork/pangenome/new_hubo_formulation/hubo_qaoa/nonvariational"
source "/nfs/users/nfs_j/jc59/quantumwork/pangenome/.venv/bin/activate"
outdir="$SCRATCH/new_hubo_formulation/nonvariational"

echo "HUBO Nonvar"
bsub -J "nonvar_hubo_$filename" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -M "$memory"\
 -o "$outdir/nonvariational.$filename.%J" -e "$outdir/error.nonvariational.$filename.%J" -G "qpg" -q "qpg" \
 "python3 $WORKING_DIR/nonvariational.py -f $filename -c $copy_numbers -n $shots"

exit 0


# dict_keys(['test_N2_W2', 'trivial', 'test_N3_W4', 'test_N4_W5', 'test_N7_W2', 'test_N7_W3', 'test_N7_W4', 'test_N8_W2', 'test_N8_W3'])