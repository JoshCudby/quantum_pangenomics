#!/bin/bash

usage()
{
    echo "usage: qubo_nonvariational_hardware.sh [[-f file -m memory -n shots -N nodes] | [-h]]"
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

WORKING_DIR="/nfs/users/nfs_j/jc59/quantumwork/pangenome/new_qubo_formulation/qubo_qaoa/nonvariational"
source "/nfs/users/nfs_j/jc59/quantumwork/pangenome/.venv/bin/activate"
outdir="$SCRATCH/new_qubo_formulation/oriented/nonvariational/hardware"

echo "QUBO Nonvar"
bsub -J "nonvar_hardware_qubo_$filename" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -M "$memory"\
 -o "$outdir/nonvar.hardware.$filename.%J" -e "$outdir/error.nonvar.hardware.$filename.%J" -G "qpg" -q "qpg" \
 "python3 $WORKING_DIR/nonvariational_hardware.py -f $filename -N $nodes -n $shots $simulation $error_mitigation"

exit 0


