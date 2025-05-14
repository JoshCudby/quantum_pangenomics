#!/bin/bash

usage()
{
    echo "usage: test_from_data.sh [[-f file -m memory -g num_gpu] | [-h]]"
}

memory="4000"
num_gpu="1"
filename="test_filename"

while [ "$1" != "" ]; do
    case $1 in
        -f | --file )           shift
                                filename="$1"
                                ;;
        -m | --memory )         shift
                                memory="$1"
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
outdir="$SCRATCH/out/prog_qaoa"

# Qiskit Testing
echo "Qiskit Testing"
bsub -J "data_$filename" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -gpu "num=$num_gpu:aff=no" -M "$memory"\
 -o "$outdir/test_data.$filename.%J" -e "$outdir/error.test_data.$filename.%J" -G "qpg" -q "qpg" \
 "python3 $WORKING_DIR/test_from_data.py -f $filename -m $memory"

exit 0
