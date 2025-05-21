#!/bin/bash

usage()
{
    echo "usage: test_cacheblocking.sh [[-m memory -b blocking] | [-h]]"
}

memory="4000"
num_gpu="1"

while [ "$1" != "" ]; do
    case $1 in
        -m | --memory )         shift
                                memory="$1"
                                ;;
        -b | --blocking )       shift
                                blocking="$1"
                                ;;
        -g | --num-gpu )        shift
                                num_gpu="$1"
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
bsub -J "cacheblocking" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -gpu "num=$num_gpu:aff=no:j_exclusive=yes" -M "$memory"\
 -o "$outdir/cacheblocking.%J" -e "$outdir/error.cacheblocking.%J" -G "qpg" -q "qpg" \
 "python3 $WORKING_DIR/test_cacheblocking.py -b $blocking -m $memory -g $num_gpu"

exit 0
