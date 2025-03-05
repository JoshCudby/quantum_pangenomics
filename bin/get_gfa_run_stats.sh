#!/bin/bash
export PATH="/nfs/users/nfs_j/jc59/quantumwork/pangenome/bin:$PATH"

mode="2"
max_seed="50"
prefix="$1"


for solver in mqlib dwave gurobi; do
    rm "$prefix.$mode.$solver.avg.txt" 2>/dev/null
    for t in 3 5 15 30 60 120; do
        rm "$mode.$solver.summary$t.txt" 2>/dev/null
        rm "$mode.$solver.max$t.txt" 2>/dev/null
    done

    for t in 3 5 15 30 60 120; do
        for i in $(seq 1 $max_seed); do cat $(printf "$mode.$solver%05d" "$i")/sim.out 2>/dev/null | grep -A 6 "Summary $t" >> "$mode.$solver.summary$t.txt"; done;
        max_gfa_sim.sh "$mode.$solver.summary$t.txt" >> "$mode.$solver.max$t.txt"
        echo "Average stats for best runs with time limit $t" >> "$prefix.$mode.$solver.avg.txt"
        average_gfa_sim.sh "$mode.$solver.max$t.txt" >> "$prefix.$mode.$solver.avg.txt"
        echo "===============" >> "$prefix.$mode.$solver.avg.txt"
    done

    cp "$prefix.$mode.$solver.avg.txt" "/nfs/users/nfs_j/jc59/quantumwork/pangenome/out"
done

solver=pathfinder
rm "$prefix.$mode.$solver.avg.txt" 2>/dev/null
rm "$mode.$solver.summary.txt" 2>/dev/null
rm "$mode.$solver.max.txt" 2>/dev/null
for i in $(seq 1 $max_seed); do cat $(printf "$mode.$solver%05d" "$max_seed")/sim.out 2>/dev/null | grep -A 6 "Summary" >> "$mode.$solver.summary.txt"; done;
max_gfa_sim.sh "$mode.$solver.summary.txt" >> "$mode.$solver.max.txt"
echo "Average stats" >> "$prefix.$mode.$solver.avg.txt"
average_gfa_sim.sh "$mode.$solver.max.txt" >> "$prefix.$mode.$solver.avg.txt"
echo "===============" >> "$prefix.$mode.$solver.avg.txt"
cp "$prefix.$mode.$solver.avg.txt" "/nfs/users/nfs_j/jc59/quantumwork/pangenome/out"
exit 0