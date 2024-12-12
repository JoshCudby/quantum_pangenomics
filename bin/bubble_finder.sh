#!/bin/bash
filepath="$1"
# TODO: add other solvers
min_size=40

if [ ! -f "$filepath" ]; then
    echo "Could not read file $filepath"
    exit 1
fi
filename=$(basename "${filepath}")
dirpath=$(dirname "${filepath}")
last_dir=$(basename "${dirpath}")

datadir=$SCRATCH/data/$last_dir

pangene_fork/k8-1.2/k8-x86_64-Linux pangene_fork/pangene.js call "$filepath" > "$datadir/$filename.bubble.txt"

declare -a bubbles

mapfile -t bubbles < <(
    awk -F'\t' '/^BB/ && $8 >= '"$min_size"' {print $0}' "$datadir/$filename.bubble.txt" | sort -t$'\t' -k8,8n
)

for bubble in "${bubbles[@]}"; do
    IFS=$'\t' read -r -a columns <<< "$bubble"
    # Check if the array contains enough columns (to avoid index errors)
    if [ "${#columns[@]}" -ge 9 ]; then
        # Access the 5th, 6th, and 9th columns
        start_node="${columns[4]}"
        end_node="${columns[5]}"
        interior_nodes="${columns[8]}"
        echo "Start: $start_node. End: $end_node"
        python3 "utils/gfa_cropper.py" "$filepath" "$start_node" "$end_node" "$interior_nodes"
    else
        echo "Warning: Skipping line with insufficient columns: $line"
    fi
done