#!/bin/bash

usage()
{
    echo "usage: nonvariational_hardware.sh [[-f file -m memory -n shots -c copy_numbers] | [-h]]"
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
        --simulation )          simulation="--simulation"
                                ;;
        --error-mitigation )    error_mitigation="--error-mitigation"
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
outdir="$SCRATCH/new_hubo_formulation/nonvariational/hardware"

echo "HUBO Nonvar"
bsub -J "nonvar_hubo_hardware_$filename" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -M "$memory"\
 -o "$outdir/nonvariational_hardware.$filename.%J" -e "$outdir/error.nonvariational_hardware.$filename.%J" -G "qpg" -q "qpg" \
 "python3 $WORKING_DIR/nonvariational_hardware.py -f $filename -c $copy_numbers -n $shots $simulation $error_mitigation"

exit 0


# dict_keys(['test_N2_W2', 'trivial', 'test_N3_W4', 'test_N4_W5', 'test_N7_W2', 'test_N7_W3', 'test_N7_W4', 'test_N8_W2', 'test_N8_W3'])