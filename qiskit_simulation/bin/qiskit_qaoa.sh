#!/bin/bash
sim_method="automatic"
use_gpu="0"
memory=4000
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
echo "Qisit Testing"
bsub -R '"select[mem>'$memory'] rusage[mem='$memory']"' -gpu "num=6:aff=no" -M "$memory"\
 -o "$outdir/qiskit.test.$sim_method.gpu$use_gpu.%J" -e "$outdir/error.qiskit.test.$sim_method.gpu$use_gpu.%J" -G "qpg" -q "qpg" \
 "python3 $WORKING_DIR/qaoa.py $filepath $sim_method $use_gpu"

exit 0
