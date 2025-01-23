#!/bin/bash
qpg=/lustre/scratch127/qpg
root_dir=$qpg/jc59/full_benchmark
out_suffix=test

# Get input sequence
seqfile=/nfs/srpipe_references/references/Treponema_pallidum/default/all/fasta/NC_016844.1.fa

# TODO: repeat N times
run_index=1
outdir="$root_dir/$out_suffix/$run_index"
mkdir -p "$outdir"

# Rotate sequence
rotated_seqfile="$outdir/rotated_input.fa"
# rotate_sequence.script $seqfile > $seqfile.rotated
# TODO: delete next line
cat $seqfile > $rotated_seqfile

# Simulate sequencing data
shredded_seqfile="$outdir/shredded.fa"
/nfs/users/nfs_j/jkb/work/quantum/shred.pl -c -s 1 -l 20000 -e 1e-5 -d 30 \
  $rotated_seqfile > $shredded_seqfile

PATH=/software/badger/opt/pangenomes/bin/:$PATH
export LD_LIBRARY_PATH=/software/badger/opt/pangenomes/lib

# Assemble into a graph
syncasm -k 301 $shredded_seqfile -o $outdir/assembled.syncasm

# Run solvers

# TODO: memory change
memory=1000

gfa_filepath=$outdir/assembled.syncasm.utg.final.gfa

bsub -J "$out_suffix.$run_index.pathfinder" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -q qpg -gpu - \
     -M "$memory" -o "$outdir/pathfinder.txt" -e "$outdir/error.pathfinder.txt" -G "qpg" \
     "$qpg/cz3/QuantumTangle/pathfinder/pathfinder $gfa_filepath"

QUBO_DIR=/nfs/users/nfs_j/jc59/quantumwork/pangenome/qubo_solvers
source $QUBO_DIR/.venv/bin/activate

bsub -J "$out_suffix.$run_index.build_qubo" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -q qpg -gpu - \
     -M "$memory" -o "$outdir/build.txt" -e "$outdir/error.build.txt" -G "qpg" \
     "python3 $QUBO_DIR/qubo_solvers/oriented_tangle/build_oriented_qubo_matrix.py $gfa_filepath $outdir"

num_jobs=10
for time_limit in 15 30 60
do
  bsub -J "$out_suffix.$run_index.mqlib[1-$num_jobs]" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -q qpg -gpu - \
       -M "$memory" -o "$outdir/mqlib.$time_limit.%I.txt" -e "$outdir/error.mqlib.txt" -G "qpg"  \
       -w "done($out_suffix.$run_index.build_qubo)" \
       "python3 $QUBO_DIR/qubo_solvers/oriented_tangle/oriented_max_path.py mqlib $gfa_filepath $time_limit $outdir"

  bsub -J "$out_suffix.$run_index.gurobi[1-$num_jobs]" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -q qpg -gpu - \
       -M "$memory" -o "$outdir/gurobi.$time_limit.%I.txt" -e "$outdir/error.gurobi.txt" -G "qpg"  \
       -w "done($out_suffix.$run_index.build_qubo)" \
       "python3 $QUBO_DIR/qubo_solvers/oriented_tangle/oriented_max_path.py gurobi $gfa_filepath $time_limit $outdir"

  # bsub -J "$out_suffix.$run_index.dwave[1-$num_jobs]" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -q qpg -gpu - \
  #      -M "$memory" -o "$outdir/dwave.$time_limit.%I.txt" -e "$outdir/error.dwave.txt" -G "qpg"  \
  #      -w "done($out_suffix.$run_index.build_qubo)" \
  #      "python3 $QUBO_DIR/qubo_solvers/oriented_tangle/oriented_max_path.py dwave $gfa_filepath $time_limit $outdir"
done


exit 0