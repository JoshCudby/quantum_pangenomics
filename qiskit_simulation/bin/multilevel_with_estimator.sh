#!/bin/bash
memory=16000
num_gpu=1


WORKING_DIR=/nfs/users/nfs_j/jc59/quantumwork/pangenome/qiskit_simulation/qiskit_qaoa/standard

outdir="$SCRATCH/out/qiskit/multilevel_estimator"
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
echo "Multi level with Qiskit Estimator Testing"

bsub -J "multi_estimator" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -gpu "num=$num_gpu:aff=no:j_exclusive=no:gmem=80000" -M "$memory"\
 -o "$outdir/$filename.%J" -e "$outdir/error.$filename.%J" -G "qpg" -q "qpg" \
 "python3 $WORKING_DIR/multilevel_with_estimator.py $filename"

exit 0