#!/bin/bash

usage()
{
    echo "usage: plot_cvar.sh [[-f file -m memory -p reps --hardware] | [-h]]"
}

memory=4000
reps=4
init="random"
alpha=0.25

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
        -i | --init )           shift
                                init="$1"
                                ;;
        -a | --alpha )          shift
                                alpha="$1"
                                ;;
        -M | --method )         shift
                                method="$1"
                                ;;
        --hardware )            hardware="--hardware"
                                ;;
        --noisy )               noisy="--noisy"
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
source /nfs/users/nfs_j/jc59/quantumwork/pangenome/qiskit_simulation/qiskit_venv/bin/activate
WORKING_DIR=/nfs/users/nfs_j/jc59/quantumwork/pangenome/qiskit_simulation/qiskit_qaoa/cvar
outdir="$SCRATCH/out/qiskit/experiments"

# Qiskit Testing
echo "Qiskit Testing"
bsub -J "plot_cvar" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -M "$memory"\
 -o "$outdir/qiskit.$filename.plot.%J" -e "$outdir/error.qiskit.$filename.plot.%J" -G "qpg" -q "qpg" \
 "python3 $WORKING_DIR/plot_cvar.py -f $filename -p $reps -n $shots --init $init -a $alpha -M $method $hardware $noisy"

exit 0
