#!/bin/bash
export PATH="/nfs/users/nfs_j/jc59/quantumwork/pangenome/bin:$PATH"

max_seed="50"
prefix="$1"
run_prefix=""
# run_prefix="test_var_"

for mode in 1 2 3; do
    for solver in mqlib dwave gurobi; do
        rm "$prefix.$mode.$run_prefix$solver.avg.txt" 2>/dev/null
        for t in 3 5 15 30 60 120; do
            rm "$mode.$run_prefix$solver.summary$t.txt" 2>/dev/null
            rm "$mode.$run_prefix$solver.max$t.txt" 2>/dev/null
        done

        for t in 3 5 15 30 60 120; do
            for i in $(seq 1 $max_seed); do cat $(printf "$mode.$run_prefix$solver%05d" "$i")/sim.out 2>/dev/null | grep -A 6 "Summary $t " >> "$mode.$run_prefix$solver.summary$t.txt"; done;
            max_gfa_sim.sh "$mode.$run_prefix$solver.summary$t.txt" >> "$mode.$run_prefix$solver.max$t.txt"
            echo "Average stats for best runs with time limit $t" >> "$prefix.$mode.$run_prefix$solver.avg.txt"
            average_gfa_sim.sh "$mode.$run_prefix$solver.max$t.txt" >> "$prefix.$mode.$run_prefix$solver.avg.txt"
            echo "===============" >> "$prefix.$mode.$run_prefix$solver.avg.txt"
        done

        cp "$prefix.$mode.$run_prefix$solver.avg.txt" "/nfs/users/nfs_j/jc59/quantumwork/pangenome/out"
    done

    solver=pathfinder
    rm "$prefix.$mode.$run_prefix$solver.avg.txt" 2>/dev/null
    rm "$mode.$run_prefix$solver.summary.txt" 2>/dev/null
    rm "$mode.$run_prefix$solver.max.txt" 2>/dev/null
    for i in $(seq 1 $max_seed); do cat $(printf "$mode.$run_prefix$solver%05d" "$i")/sim.out 2>/dev/null | grep -A 6 "Summary" >> "$mode.$run_prefix$solver.summary.txt"; done;
    echo "Average stats" >> "$prefix.$mode.$run_prefix$solver.avg.txt"
    average_gfa_sim.sh "$mode.$run_prefix$solver.summary.txt" >> "$prefix.$mode.$run_prefix$solver.avg.txt"
    echo "===============" >> "$prefix.$mode.$run_prefix$solver.avg.txt"
    cp "$prefix.$mode.$run_prefix$solver.avg.txt" "/nfs/users/nfs_j/jc59/quantumwork/pangenome/out"
done
exit 0