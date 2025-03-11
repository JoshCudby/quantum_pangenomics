#!/bin/bash
export PATH="/nfs/users/nfs_j/jc59/quantumwork/pangenome/bin:$PATH"

mode="2"
seed="6"
prefix="$1"
run_prefix=""
# run_prefix="test_var_"


for solver in mqlib dwave gurobi; do
    file_name_base=$(printf "$mode.$run_prefix$solver.%05d" "$seed")

    rm "$file_name_base.avg.txt" 2>/dev/null
    for t in 3 5 15 30 60 120; do
        rm "$file_name_base.summary$t.txt" 2>/dev/null
        rm "$file_name_base.max$t.txt" 2>/dev/null
    done

    for t in 3 5 15 30 60 120; do
        cat $(printf "$mode.$run_prefix$solver%05d" "$seed")/sim.out 2>/dev/null | grep -A 6 "Summary $t " >> "$file_name_base.summary$t.txt"
        max_gfa_sim.sh "$file_name_base.summary$t.txt" >> "$file_name_base.max$t.txt"
        echo "Average stats for best runs with time limit $t" >> "$file_name_base.avg.txt"
        average_gfa_sim.sh "$file_name_base.max$t.txt" >> "$file_name_base.avg.txt"
        echo "===============" >> "$file_name_base.avg.txt"
    done

    cp "$file_name_base.avg.txt" "/nfs/users/nfs_j/jc59/quantumwork/pangenome/out"
done


solver=pathfinder
file_name_base=$(printf "$mode.$run_prefix$solver.%05d" "$seed")

rm "$file_name_base.avg.txt" 2>/dev/null
rm "$file_name_base.summary.txt" 2>/dev/null
rm "$file_name_base.max.txt" 2>/dev/null
cat $(printf "$mode.$run_prefix$solver%05d" "$seed")/sim.out 2>/dev/null | grep -A 6 "Summary" >> "$file_name_base.summary.txt"
max_gfa_sim.sh "$file_name_base.summary.txt" >> "$file_name_base.max.txt"
echo "Average stats" >> "$file_name_base.avg.txt"
average_gfa_sim.sh "$file_name_base.max.txt" >> "$file_name_base.avg.txt"
echo "===============" >> "$file_name_base.avg.txt"
cp "$file_name_base.avg.txt" "/nfs/users/nfs_j/jc59/quantumwork/pangenome/out"

exit 0