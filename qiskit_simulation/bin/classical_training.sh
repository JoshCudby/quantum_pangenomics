#!/bin/bash
usage()
{
    echo "usage: classical_training.sh [[-f file -p reps] | [-h]]"
}
reps=4
memory=4000

while [ "$1" != "" ]; do
    case $1 in
        -f | --file )           shift
                                filename="$1"
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
WORKING_DIR=/nfs/users/nfs_j/jc59/quantumwork/pangenome/qiskit_simulation/qiskit_qaoa/parameter_transfer
outdir="$SCRATCH/out/training"

echo "Qiskit Testing"
bsub -J "training.$filename.$reps" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -gpu - -M "$memory"\
 -o "$outdir/training.$filename.p$reps.%J.out" -e "$outdir/training.$filename.p$reps.%J.err" -G "qpg" -q "qpg" \
 "python3 $WORKING_DIR/classical_training.py -f $filename -p $reps"

exit 0
