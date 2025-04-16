#!/bin/bash

PATH=/software/badger/opt/pangenomes/bin/:/software/sciops/pkgg/bwa/0.7.17/bin:$PATH
export LD_LIBRARY_PATH=/software/badger/opt/pangenomes/lib
mode=1
num_training=1

min=${min:-"1"}
max=${max:-"4"}
do_small=${do_small:-"0"}

# TODO: why need to export this before?
export QDIR=${QDIR:-$(pwd)}
export PATH=$QDIR:$PATH

rm -rf sim"$min$max$do_small".out
rm -rf sim"$min$max$do_small".err

datadir=/lustre/scratch127/qpg/jc59/data
QUBO_DIR=/nfs/users/nfs_j/jc59/quantumwork/pangenome/qubo_solvers

(
if [[ do_small -eq 1 ]]; then
    for file in trivial small_test test_N4_W5 test_N4_W6; do
    # for file in trivial; do
        cd /lustre/scratch127/qpg/jc59/qaoa_depth_width || exit
        out_dir=$file
        echo "$out_dir"
        rm -rf "$out_dir" 2>/dev/null
        mkdir "$out_dir"
        cd "$out_dir" || exit
        copy_numbers=$(perl -e '
        use strict;
        while (<>) {
            next unless /^S/;
            m/SC:f:([0-9.]*)/;
            print int($1), ",";
        }
        ' $datadir/$file.gfa) 

        source $QUBO_DIR/.venv/bin/activate 
        python3 $QUBO_DIR/qubo_solvers/oriented_tangle/build_oriented_qubo_matrix.py -f $datadir/$file.gfa -d "." -c $copy_numbers -p "10,5,1"

        source /lustre/scratch127/qpg/jc59/.venv/qiskit_venv/bin/activate
        python3 /nfs/users/nfs_j/jc59/quantumwork/pangenome/qiskit_simulation/qiskit_qaoa/cvar/qaoa_depth_width.py -f $file.gfa -d "."
    done
fi 

for mode in $(seq 0 0); do
    . ${CONFIG:-$QDIR/sim_path_config_hifi_mg.sh} "$mode"

    for seed in $(seq "$min" "$max"); do
        cd /lustre/scratch127/qpg/jc59/qaoa_depth_width || exit
        out_dir=$(printf "$mode.%05d" "$seed")
        echo "$out_dir"
        rm -rf "$out_dir" 2>/dev/null
        mkdir "$out_dir"
        cd "$out_dir" || exit

        k1=75
        k2=50
        k3=35


        # Create fake genomes and use the training set to create a pangenome
        # Creates:
        #    pop.gfa
        #    pop.gfa.ns$k1
        #    pop.gfa.ns$k2
        #    pop.gfa.ns$k3
        #    fofn.test
        #    fofn.train
        run_sim_create_gfa.sh "$seed" $k1 $k2 $k3 "$mode" $num_training

        # Foreach test genome, not used in pangenome creation, find path and eval
        for i in $(cat fofn.test)
        do
            if [ "x$use_mg" = "x1" ]
            then
                # Add weights to the GFA via minigraph
                # Creates:
                #     $i.gfa (primary output; annotated pop.gfa)
                #     $i.shred.fa
                #     $i.mg
                run_sim_add_gfa_weights_mg.sh pop.gfa "$i" "$mode"
            else
                # Add weights to the GFA via kmer2node.
                # Creates:
                #     $i.gfa (primary output; annotated pop.gfa)
                #     $i.shred.fa
                #     $i.nodes.$k1
                #     $i.nodes.$k2
                #     $i.nodes.$k3
                #     $i.nodes
                run_sim_add_gfa_weights.sh pop.gfa $i $k1 $k2 $k3 $mode
            fi

            copy_numbers=$(perl -e '
            use strict;
            while (<>) {
                next unless /^S/;
                m/dc:f:([0-9.]*)/;
                print int($1/'$shred_depth' + .8), ",";
            }
            ' $i.gfa)

            echo $copy_numbers >> sim.err

            source $QUBO_DIR/.venv/bin/activate 
            python3 $QUBO_DIR/qubo_solvers/oriented_tangle/build_oriented_qubo_matrix.py -f $i.gfa -d "./" -c $copy_numbers -p "10,5,1"

            source /lustre/scratch127/qpg/jc59/.venv/qiskit_venv/bin/activate
            python3 /nfs/users/nfs_j/jc59/quantumwork/pangenome/qiskit_simulation/qiskit_qaoa/cvar/qaoa_depth_width.py -f $i.gfa -d "./"
        done

    done
done

) 2>sim"$min$max$do_small".err | tee sim"$min$max$do_small".out

exit 0