#!/bin/bash

usage()
{
    echo "usage: plot_cvar.sh [[-f file -m memory -p reps --hardware] | [-h]]"
}

memory=4000
reps=4
init="ramp"

while [ "$1" != "" ]; do
    case $1 in
        -f | --file )           shift
                                filename="$1"
                                ;;
        -m | --memory )         shift
                                memory="$1"
                                ;;
        -M | --method )         shift
                                method="$1"
                                ;;
        -p | --reps )           shift
                                reps="$1"
                                ;;
        -n | --shots )          shift
                                shots="$1"
                                ;;
        --init )                shift
                                init="$1"
                                ;;
        -i )                    shift
                                maxiter="$1"
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

# module load cuda-12.1.1
# LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
# module load ISG/experimental/fg12/openmpi/5.0.4-cuda12.1-lsf
WORKING_DIR=/nfs/users/nfs_j/jc59/quantumwork/pangenome/prog_qaoa/qiskit_prog_qaoa
outdir="$SCRATCH/out/prog_qaoa/tangle"

# Qiskit Testing
echo "Qiskit Testing"
bsub -J "plot_prog_qaoa" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -M "$memory"\
 -o "$outdir/plot.$filename.%J" -e "$outdir/error.plot.$filename.%J" -G "qpg" -q "normal" \
 "python3 $WORKING_DIR/plot_prog_qaoa.py -f $filename -p $reps -n $shots --init $init -i $maxiter -M $method"

exit 0
