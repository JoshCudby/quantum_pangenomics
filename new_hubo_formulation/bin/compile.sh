#!/bin/bash

usage()
{
    echo "usage: compile_hubo.sh [[-f file -m memory -p reps -n shots -g num_gpu -i init -R rows -C cols -d swap_depth] | [-h]]"
}

memory="4000"
timeout=60

while [ "$1" != "" ]; do
    case $1 in
        -f | --file )           shift
                                filename="$1"
                                ;;
        -m | --memory )         shift
                                memory="$1"
                                ;;
        -t | --timeout )        shift
                                timeout="$1"
                                ;;
        -k | --keep )           shift
                                keep="$1"
                                ;;
        -c | --copy-numbers )   shift
                                copy_numbers="$1"
                                ;;
        -C | --coupling )       shift
                                coupling="$1"
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

WORKING_DIR="/nfs/users/nfs_j/jc59/quantumwork/pangenome/new_hubo_formulation/hubo_qaoa"
source "/nfs/users/nfs_j/jc59/quantumwork/pangenome/.venv/bin/activate"
outdir="$SCRATCH/new_hubo_formulation"

echo "HUBO Testing"
bsub -J "optimize_hubo" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -gpu "num=1:aff=no:j_exclusive=yes" -M "$memory"\
 -o "$outdir/$filename.%J" -e "$outdir/error.$filename.%J" -G "qpg" -q "qpg" \
 "python3 $WORKING_DIR/compilation.py -f $filename --fraction-four 0 --fraction-six 1 \
  --times-to-keep $keep -t $timeout -e 1 -c $copy_numbers -C $coupling"

exit 0


