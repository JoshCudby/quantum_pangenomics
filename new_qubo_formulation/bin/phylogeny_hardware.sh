#!/bin/bash



memory="4000"
shots=4000

while [ "$1" != "" ]; do
    case $1 in
        -v | --vertices )       shift
                                vertices="$1"
                                ;;
        -m | --memory )         shift
                                memory="$1"
                                ;;
        -n | --shots )          shift
                                shots="$1"
                                ;;
        --simulation )          simulation="--simulation"
                                ;;
        --error-mitigation )    error_mitigation="--error-mitigation"
                                ;;
        * )                     exit 1
    esac
    shift
done

## MAIN

WORKING_DIR="/nfs/users/nfs_j/jc59/quantumwork/pangenome/new_qubo_formulation/qubo_qaoa/nonvariational/phylogeny"
source "/nfs/users/nfs_j/jc59/quantumwork/pangenome/.venv/bin/activate"
outdir="$SCRATCH/phylogeny"

echo "QUBO Nonvar"
bsub -J "nonvar_phylo_$vertices" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -M "$memory"\
 -o "$outdir/nonvar.hardware.$vertices.%J" -e "$outdir/error.nonvar.hardware.$vertices.%J" -G "qpg" -q "qpg" \
 "python3 $WORKING_DIR/nonvariational_phylo_hardware.py -v $vertices -n $shots $simulation $error_mitigation"

exit 0


