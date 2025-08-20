#!/bin/bash

usage()
{
    echo "usage: optimize_hubo.sh [[-f file -m memory -p reps -n shots -g num_gpu -i init -R rows -C cols -d swap_depth] | [-h]]"
}

memory="4000"
reps="4"
num_gpu="1"
init="random"
shots=2000

while [ "$1" != "" ]; do
    case $1 in
        -f | --file )           shift
                                filename="$1"
                                ;;
        -m | --memory )         shift
                                memory="$1"
                                ;;
        -p | --reps )           shift
                                reps="$1"
                                ;;
        -n | --shots )          shift
                                shots="$1"
                                ;;
        -g | --num-gpu )        shift
                                num_gpu="$1"
                                ;;
        -i | --init )           shift
                                init="$1"
                                ;;
        -R | --rows )           shift
                                rows="$1"
                                ;;
        -C | --cols )           shift
                                cols="$1"
                                ;;
        -d | --depth )          shift
                                swap_depth="$1"
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
outdir="$SCRATCH/hubo/order4"

echo "HUBO Testing"
bsub -J "optimize_hubo" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -gpu "num=$num_gpu:aff=no" -M "$memory"\
 -o "$outdir/$filename.%J" -e "$outdir/error.$filename.%J" -G "qpg" -q "qpg" \
 "python3 $WORKING_DIR/hubo_optimisation.py -f $filename -p $reps -m $memory -n $shots --init $init -d $swap_depth -R $rows -C $cols -e 0"

exit 0


