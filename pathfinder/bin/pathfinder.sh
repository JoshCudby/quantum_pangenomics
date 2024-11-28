#!/bin/bash

filepath="$1"
filename=$(basename -- "$filepath")

rm ./out/$filename.txt
rm ./out/error.$filename.txt

bsub -o ./out/$filename.txt -e ./out/error.$filename.txt\
 -n 4\
 -R "select[mem>32000] rusage[mem=32000]" -M 32000 -q qpg -gpu -\
 "/lustre/scratch127/qpg/cz3/QuantumTangle/pathfinder/pathfinder $filepath -N 100000000"

exit 0