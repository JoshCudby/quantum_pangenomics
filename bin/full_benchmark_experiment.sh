#!/bin/bash
qpg=/lustre/scratch127/qpg
root_dir=$qpg/jc59/full_benchmark

kmer="$1"
dt="$2"

out_suffix=$kmer.$dt


# QUBO venv
QUBO_DIR=/nfs/users/nfs_j/jc59/quantumwork/pangenome/qubo_solvers
source $QUBO_DIR/.venv/bin/activate


# TODO: memory change?
memory=2000
num_jobs=5


# Get input sequence
seqfile=/nfs/srpipe_references/references/Treponema_pallidum/default/all/fasta/NC_016844.1.fa

for run in {1..5}; do
     seed=$RANDOM
     outdir="$root_dir/$out_suffix/$run"
     mkdir -p "$outdir"


     # Rotate sequence
     rotated_seqfile="$outdir/rotated_input.fa"
     /nfs/users/nfs_j/jc59/quantumwork/pangenome/bin/rotate_sequence.pl $seed $seqfile > $rotated_seqfile


     # Simulate sequencing data
     shredded_seqfile="$outdir/shredded.fa"
     /nfs/users/nfs_j/jkb/work/quantum/shred.pl -s 1 -l 20000 -e 0.001 -d 30 $rotated_seqfile > $shredded_seqfile


     # Assemble into a graph
     /nfs/users/nfs_j/jc59/quantumwork/pangenome/modules/oatk/syncasm -k $kmer $shredded_seqfile -o "$outdir/assembled.syncasm" &> "$outdir/error.syncasm"


     # Run solvers
     gfa_filepath=$outdir/assembled.syncasm.utg.final.gfa

     # Pathfinder
     bsub -J "$out_suffix.$run.pathfinder" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -q normal \
          -M "$memory" -o "$outdir/pathfinder.txt" -e "$outdir/error.pathfinder.txt" -G "qpg" \
          "$qpg/cz3/QuantumTangle/pathfinder/pathfinder -a $gfa_filepath"


     # Only 2 gurobi sessions at any time
     if [[ $run -lt 3 ]]; then
          depend_cond=""
     else
          let prev_run="$run-2"
          depend_cond="numended($out_suffix.$prev_run.gurobi,*)"
     fi

     bsub -J "$out_suffix.$run.build_qubo" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -q normal \
          -M "$memory" -o "$outdir/build.txt" -e "$outdir/error.build.txt" -G "qpg" \
          -w "$depend_cond" \
          "python3 $QUBO_DIR/qubo_solvers/oriented_tangle/build_oriented_qubo_matrix.py $gfa_filepath $outdir"

     
     for time_limit in 5 15 30 60; do
          bsub -J "$out_suffix.$run.mqlib[1-$num_jobs]" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -q normal \
               -M "$memory" -o "$outdir/mqlib.$time_limit.%I.txt" -e "$outdir/error.mqlib.txt" -G "qpg"  \
               -w "done($out_suffix.$run.build_qubo)" \
               "python3 $QUBO_DIR/qubo_solvers/oriented_tangle/oriented_max_path.py mqlib $gfa_filepath $time_limit $outdir"

          bsub -J "$out_suffix.$run.gurobi[1-$num_jobs]" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -q normal \
               -M "$memory" -o "$outdir/gurobi.$time_limit.%I.txt" -e "$outdir/error.gurobi.txt" -G "qpg"  \
               -w "done($out_suffix.$run.build_qubo)" \
               "python3 $QUBO_DIR/qubo_solvers/oriented_tangle/oriented_max_path.py gurobi $gfa_filepath $time_limit $outdir"

          bsub -J "$out_suffix.$run.dwave[1-$num_jobs]" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -q normal \
               -M "$memory" -o "$outdir/dwave.$time_limit.%I.txt" -e "$outdir/error.dwave.txt" -G "qpg"  \
               -w "done($out_suffix.$run.build_qubo)" \
               "python3 $QUBO_DIR/qubo_solvers/oriented_tangle/oriented_max_path.py dwave $gfa_filepath $time_limit $outdir"
     done
done

exit 0