#!/bin/sh
qpg=/lustre/scratch127/qpg
root_dir=$qpg/jc59/full_benchmark
out_suffix=test

# Get input sequence
seqfile=/nfs/srpipe_references/references/Treponema_pallidum/default/all/fasta/NC_016844.1.fa

run_index=1
outdir="$root_dir/$out_suffix/$run_index"
mkdir -p "$outdir"

# Rotate sequence
rotated_seqfile="$outdir/rotated_input.fa"
# rotate_sequence.script $seqfile > $seqfile.rotated
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
bsub -J "$out_suffix.$run_index.pathfinder" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -q qpg -gpu - \
     -M "$memory" -o "$outdir/pathfinder.txt" -e "$outdir/error.pathfinder.txt"\
     -G "qpg" "$qpg/cz3/QuantumTangle/pathfinder/pathfinder $outdir/assembled.syncasm.utg.final.gfa"

QUBO_DIR=/nfs/users/nfs_j/jc59/quantumwork/pangenome/qubo_solvers
source ~/.venv/qubo/bin/activate

bsub -J "$out_suffix.$run_index.build_qubo" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -M "$memory" -G "qpg" \
 -o "$outdir/build.txt" -e "$outdir/error.build.txt" -q qpg -gpu - \
 "python3 $QUBO_DIR/qubo_solvers/oriented_tangle/build_oriented_qubo_matrix.py $filepath"

# TODO: build will output to wrong dir??
# TODO: multiple timelimits
time_limit=10
bsub -J "$out_suffix.$run_index.mqlib" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -M "$memory" -G "qpg" \
 -o "$outdir/build.txt" -e "$outdir/error.build.txt" -q qpg -gpu - \
 -w "done($out_suffix.$run_index.build_qubo)" \
 "python3 $QUBO_DIR/qubo_solvers/oriented_tangle/oriented_max_path.py mqlib $filepath $time_limit"