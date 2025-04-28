#!/bin/bash

usage()
{
    echo "usage: optimize.sh [[-f file -m memory -p reps -n shots -g num_gpu -i init] | [-h]]"
}

memory="4000"
reps="4"
num_gpu="1"
init="ramp"
shots=1000
maxiter=100
filename="test_filename"

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
        -i )                    shift
                                maxiter="$1"
                                ;;
        --init )                shift
                                init="$1"
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
WORKING_DIR="/nfs/users/nfs_j/jc59/quantumwork/pangenome/prog_qaoa/qiskit_prog_qaoa"
outdir="$SCRATCH/out/prog_qaoa"

# Qiskit Testing
echo "Qiskit Testing"
bsub -J "optimize_$filename" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -gpu "num=$num_gpu:aff=no" -M "$memory"\
 -o "$outdir/$filename.%J" -e "$outdir/error.$filename.%J" -G "qpg" -q "qpg" \
 "python3 $WORKING_DIR/optimize_prog_qaoa.py -f $filename -p $reps -m $memory -n $shots -i $maxiter --init $init"

exit 0
