#!/bin/bash
memory=64000
num_gpu=1

usage()
{
    echo "usage: multilevel_tangle_qaoa.sh [[-f file] | [-h]]"
}

WORKING_DIR=/nfs/users/nfs_j/jc59/quantumwork/pangenome/openqaoa_simulation/openqaoa_tangle

outdir="$SCRATCH/out/openqaoa"
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
echo "OpenQAOA Testing"
bsub -J "openqaoa" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -gpu - -M "$memory"\
 -o "$outdir/multilevel.$filename.%J" -e "$outdir/error.multilevel.$filename.%J" -G "qpg" -q "qpg" \
 "python3 $WORKING_DIR/multilevel_qaoa.py $filename"

exit 0