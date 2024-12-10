#!/bin/bash

filepath="$1"
filename=$(basename -- "$filepath")

rm $SCRATCH/out/pathfinder/$filename.txt
rm $SCRATCH/out/pathfinder/error.$filename.txt

bsub -o $SCRATCH/out/pathfinder/$filename.txt -e $SCRATCH/out/pathfinder/error.$filename.txt\
 -n 4\
 -R "select[mem>32000] rusage[mem=32000]" -M 32000 -q qpg -gpu -\
 "/lustre/scratch127/qpg/cz3/QuantumTangle/pathfinder/pathfinder $filepath -N 100000000"

exit 0