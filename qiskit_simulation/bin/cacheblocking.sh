#!/bin/bash
memory=4000
num_gpu=3
qubits=30
while [ "$1" != "" ]; do
    case $1 in
        -m | --memory )         shift
                                memory="$1"
                                ;;
        -n | --num-gpu )        shift
                                num_gpu="$1"
                                ;;
        -q | --qubits )         shift
                                qubits="$1"
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
WORKING_DIR=/nfs/users/nfs_j/jc59/quantumwork/pangenome/qiskit_simulation/qiskit_qaoa/standard
outdir="$SCRATCH/out/qiskit"

# Qiskit Testing
echo "Qiskit Testing"
bsub -J "cacheblocking$memory$num_gpu" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -gpu "num=$num_gpu:aff=no:j_exclusive=no" -M "$memory"\
 -o "$outdir/qiskit.cacheblocking.qubits$qubits.mem$memory.num$num_gpu.%J" -e "$outdir/error.qiskit.cacheblocking.qubits$qubits.mem$memory.num$num_gpu.%J" -G "qpg" -q "qpg" \
 "python3 $WORKING_DIR/cacheblocking.py $qubits"

exit 0
