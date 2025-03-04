#!/bin/bash
export PATH="/nfs/users/nfs_j/jc59/quantumwork/pangenome/bin:$PATH"

mode="1"
max_seed="20"

for solver in mqlib dwave gurobi; do
    rm "$mode.$solver.avg.txt" 2>/dev/null
    for t in 30 60 120; do
        rm "$mode.$solver.summary$t.txt" 2>/dev/null
        rm "$mode.$solver.max$t.txt" 2>/dev/null
    done

    for t in 30 60 120; do
        for i in $(seq 1 $max_seed); do cat 1.$solver.$i.log.txt 2>/dev/null | grep -A 6 "Summary $t" >> "$mode.$solver.summary$t.txt"; done;
        max_gfa_sim.sh "$mode.$solver.summary$t.txt" >> "$mode.$solver.max$t.txt"
        echo "Average stats for best runs with time limit $t" >> "$mode.$solver.avg.txt"
        average_gfa_sim.sh "$mode.$solver.max$t.txt" >> "$mode.$solver.avg.txt"
        echo "===============" >> "$mode.$solver.avg.txt"
    done

    cp "$mode.$solver.avg.txt" "/nfs/users/nfs_j/jc59/quantumwork/pangenome/out"
done

exit 0