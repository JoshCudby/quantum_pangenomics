#!/bin/bash
memory=64000
num_gpu=1

usage()
{
    echo "usage: multilevel_experiment.sh [[-f file] | [-h]]"
}


WORKING_DIR=/nfs/users/nfs_j/jc59/quantumwork/pangenome/qiskit_simulation/qiskit_qaoa/standard

outdir="$SCRATCH/out/qiskit/multilevel"
mkdir -p $outdir

while [ "$1" != "" ]; do
    case $1 in
        -f | --file )           shift
                                filename="$1"
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done

# Qiskit Testing
echo "Multi level Qiskit Testing"
bsub -J "multi_level" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -gpu "num=$num_gpu:aff=no:j_exclusive=no:gmem=80000" -M "$memory"\
 -o "$outdir/$filename.%J" -e "$outdir/error.$filename.%J" -G "qpg" -q "qpg" \
 "python3 $WORKING_DIR/multi_level_experiment.py $filename"

exit 0