#!/bin/bash

usage()
{
    echo "usage: optimize_cvar.sh [[-f file -m memory -p reps -n shots -g num_gpu -i init --hardware --noisy] | [-h]]"
}

memory="4000"
reps="4"
num_gpu="1"
init="random"
shots=2000

while [ "$1" != "" ]; do
    case $1 in
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
        --hardware )            hardware="--hardware"
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
WORKING_DIR="/nfs/users/nfs_j/jc59/quantumwork/pangenome/qiskit_simulation/qiskit_qaoa/cvar"
outdir="$SCRATCH/out/orson"

# Qiskit Testing
echo "Qiskit Testing"
bsub -J "optimize_cvar" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -gpu "num=$num_gpu:aff=no" -M "$memory"\
 -o "$outdir/phylo.%J" -e "$outdir/error.phylo.%J" -G "qpg" -q "qpg" \
 "python3 $WORKING_DIR/optimize2.py -f "NULL_FILE" -p $reps -m $memory -n $shots --init $init $hardware"

exit 0
