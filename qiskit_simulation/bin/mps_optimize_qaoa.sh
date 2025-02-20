#!/bin/bash
memory=16000
reps=4
while [ "$1" != "" ]; do
    case $1 in
        -f | --file )           shift
                                filepath="$1"
                                ;;
        -m | --memory )         shift
                                memory="$1"
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
filename=$(basename "${filepath}")
seed=${1:-$$}

## MAIN

# module load cuda-12.1.1
# LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
# module load ISG/experimental/fg12/openmpi/5.0.4-cuda12.1-lsf
WORKING_DIR=/nfs/users/nfs_j/jc59/quantumwork/pangenome/qiskit_simulation/qiskit_qaoa/standard
outdir="$SCRATCH/out/qiskit"

# Qiskit Testing
echo "Qiskit Testing"
bsub -J "mps" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -gpu - -M "$memory"\
 -o "$outdir/qiskit.$filename.mps.$seed" -e "$outdir/error.qiskit.$filename.mps.$seed" -G "qpg" -q "qpg" \
 "python3 $WORKING_DIR/mps_optimize_qaoa.py $seed $filepath $reps $memory scipy"

exit 0
