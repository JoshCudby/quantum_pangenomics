#!/bin/bash
sim_method="automatic"
use_gpu="0"
memory=4000
reps=4
num_gpu=4
while [ "$1" != "" ]; do
    case $1 in
        -f | --file )           shift
                                filepath="$1"
                                ;;
        -m | --memory )         shift
                                memory="$1"
                                ;;
        -s | --sim-method )     shift
                                sim_method="$1"
                                ;;
        -g | --gpu )            shift
                                use_gpu="$1"
                                ;;
        -n | --num-gpu )        shift
                                num_gpu="$1"
                                ;;
        -p | --reps )           shift
                                reps="$1"
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
WORKING_DIR=/nfs/users/nfs_j/jc59/quantumwork/pangenome/qiskit_simulation/qiskit_qaoa
outdir="$SCRATCH/out/qiskit"

# Qiskit Testing
echo "Qiskit Testing"
bsub -J "$sim_method.$use_gpu" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -gpu "num=$num_gpu:aff=no:j_exclusive=yes" -M "$memory"\
 -o "$outdir/qiskit.test.$sim_method.gpu$use_gpu.num$num_gpu.%J" -e "$outdir/error.qiskit.test.$sim_method.gpu$use_gpu.num$num_gpu.%J" -G "qpg" -q "qpg" \
 "python3 $WORKING_DIR/qaoa.py $filepath $sim_method $reps $use_gpu $memory scikit"

exit 0
