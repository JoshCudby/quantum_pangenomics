#!/bin/bash

usage()
{
    echo "usage: hubo_no_noise.sh [[-f file -m memory -p reps -n shots] | [-h]]"
}

memory="4000"
reps="1"

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
        -c | --copy_numbers )   shift
                                copy_numbers="$1"
                                ;;
        -N )                    shift
                                nodes="$1"
                                ;;
        -T )                    shift
                                time="$1"
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
WORKING_DIR="/nfs/users/nfs_j/jc59/quantumwork/pangenome/qiskit_simulation/qiskit_qaoa/hubo"
source "/nfs/users/nfs_j/jc59/quantumwork/pangenome/qiskit_simulation/qiskit_venv/bin/activate"
outdir="$SCRATCH/out/qiskit/hubo_no_shot_noise"

# Qiskit Testing
echo "Qiskit Testing"
bsub -J "hubo_no_noise.$filename.$reps" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -gpu "num=1:aff=no:j_exclusive=yes" -M "$memory"\
 -o "$outdir/$filename.p$reps.%J" -e "$outdir/error.$filename.p$reps.%J" -G "qpg" -q "qpg" \
 "python3 $WORKING_DIR/hubo_optimisation_no_noise_all_to_all.py -f $filename -p $reps -m $memory -N $nodes -T $time -c $copy_numbers"

exit 0
