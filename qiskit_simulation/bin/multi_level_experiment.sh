#!/bin/bash
memory=32000
num_gpu=1


WORKING_DIR=/nfs/users/nfs_j/jc59/quantumwork/pangenome/qiskit_simulation/qiskit_qaoa

outdir="$SCRATCH/out/qiskit/multilevel"
mkdir -p $outdir

while [ "$1" != "" ]; do
    case $1 in
        -f | --file )           shift
                                filepath="$1"
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

# Qiskit Testing
echo "Multi level Qiskit Testing"
bsub -J "multi_level" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -gpu "num=$num_gpu:aff=no:j_exclusive=yes" -M "$memory"\
 -o "$outdir/$filename.%J" -e "$outdir/error.$filename.%J" -G "qpg" -q "qpg" \
 "python3 $WORKING_DIR/multi_level_experiment.py $filepath"

exit 0