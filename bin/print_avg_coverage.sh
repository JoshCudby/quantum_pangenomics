#!/bin/bash
export PATH="/nfs/users/nfs_j/jc59/quantumwork/pangenome/bin:$PATH"

max_seed="1000"
# run_prefix="mod_penalty_"
run_prefix="eval"
mode=2

for solver in mqlib gurobi; do
    for t in 3 30; do
        rm "$mode.$run_prefix.$solver.coverages$t.txt" 2>/dev/null
        rm "$mode.$run_prefix.$solver.summary$t.txt" 2>/dev/null
    done

    for t in 3 30; do
        for i in $(seq 1 $max_seed); do cat $mode.$run_prefix.$solver.$i.00002/sim.out 2>/dev/null | grep -A 2 "Summary $t " >> "$mode.$run_prefix.$solver.summary$t.txt"; done;
        awk 'NF { 
            if (NF >= 5 && $4 ~ /^[0-9]+(\.[0-9]+)?%$/ && $5 ~ /^[0-9]+(\.[0-9]+)?%$/) {
                gsub(/%/, "", $4)
                gsub(/%/, "", $5)
                avg = ($4 + $5) / 2
                print avg
            }
        }' "$mode.$run_prefix.$solver.summary$t.txt" > "$mode.$run_prefix.$solver.coverages$t.txt"

        cp "$mode.$run_prefix.$solver.coverages$t.txt" "/nfs/users/nfs_j/jc59/quantumwork/pangenome/out"
    done

    for t in 3 30; do
        awk 'NF { 
            if (NF >= 7 && $7 ~ /^[0-9]+(\.[0-9]+)?$/) {
                print $7
            }
        }' "$mode.$run_prefix.$solver.summary$t.txt" > "$mode.$run_prefix.$solver.breaks$t.txt"

        cp "$mode.$run_prefix.$solver.breaks$t.txt" "/nfs/users/nfs_j/jc59/quantumwork/pangenome/out"
    done    
done

exit 0