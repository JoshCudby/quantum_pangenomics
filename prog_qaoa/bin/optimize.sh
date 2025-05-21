#!/bin/bash

usage()
{
    echo "usage: optimize.sh [[-f file -m memory -M method -b blocking -p reps -n shots -g num_gpu -i max_iter --init] | [-h]]"
}

memory="4000"
reps="4"
num_gpu="1"
init="ramp"
shots=1000
maxiter=100
filename="test_filename"
method="Powell"
blocking="30"

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
        -g | --num-gpu )        shift
                                num_gpu="$1"
                                ;;
        -b | --blocking )       shift
                                blocking="$1"
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
outdir="$SCRATCH/out/prog_qaoa/tangle"

# Qiskit Testing
echo "Qiskit Testing"
bsub -J "optimize_$filename" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -gpu "num=$num_gpu:aff=no:j_exclusive=yes" -M "$memory"\
 -o "$outdir/$filename.g$num_gpu.cacheblocking$blocking.%J" -e "$outdir/error.$filename.g$num_gpu.cacheblocking$blocking.%J" -G "qpg" -q "qpg" \
 "python3 $WORKING_DIR/optimize_prog_qaoa.py -f $filename -p $reps -b $blocking -m $memory -n $shots -i $maxiter --init $init -M $method"

exit 0
