#!/bin/bash
qpg=/lustre/scratch127/qpg
root_dir=$qpg/jc59/full_benchmark
out_suffix=test

# Get input sequence
seqfile=/nfs/srpipe_references/references/Treponema_pallidum/default/all/fasta/NC_016844.1.fa

# TODO: repeat N times
seed=${1:-$$}
outdir="$root_dir/$out_suffix/$seed"
mkdir -p "$outdir"

# Rotate sequence
rotated_seqfile="$outdir/rotated_input.fa"
/nfs/users/nfs_j/jc59/quantumwork/pangenome/bin/rotate_sequence.pl $seed $seqfile > $rotated_seqfile


# Simulate sequencing data
shredded_seqfile="$outdir/shredded.fa"
/nfs/users/nfs_j/jkb/work/quantum/shred.pl -c -s 1 -l 20000 -e 1e-5 -d 30 \
  $rotated_seqfile > $shredded_seqfile

PATH=/software/badger/opt/pangenomes/bin/:$PATH
export LD_LIBRARY_PATH=/software/badger/opt/pangenomes/lib

# Assemble into a graph
syncasm -k 301 $shredded_seqfile -o $outdir/assembled.syncasm >> $outdir/error.syncasm

# Run solvers

# TODO: memory change
memory=1000

gfa_filepath=$outdir/assembled.syncasm.utg.final.gfa
qubo_data_filepath=$outdir/qubo_data_assembled.syncasm.utg.final.gfa.npy

# Pathfinder
bsub -J "$out_suffix.$seed.pathfinder" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -q qpg -gpu - \
     -M "$memory" -o "$outdir/pathfinder.txt" -e "$outdir/error.pathfinder.txt" -G "qpg" \
     "$qpg/cz3/QuantumTangle/pathfinder/pathfinder $gfa_filepath"

# QUBO
QUBO_DIR=/nfs/users/nfs_j/jc59/quantumwork/pangenome/qubo_solvers
source $QUBO_DIR/.venv/bin/activate

bsub -J "$out_suffix.$seed.build_qubo" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -q qpg -gpu - \
     -M "$memory" -o "$outdir/build.txt" -e "$outdir/error.build.txt" -G "qpg" \
     "python3 $QUBO_DIR/qubo_solvers/oriented_tangle/build_oriented_qubo_matrix.py $gfa_filepath $outdir"

num_jobs=1
for time_limit in 15
do
  bsub -J "$out_suffix.$seed.mqlib[1-$num_jobs]" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -q qpg -gpu - \
       -M "$memory" -o "$outdir/mqlib.$time_limit.%I.txt" -e "$outdir/error.mqlib.txt" -G "qpg"  \
       -w "done($out_suffix.$seed.build_qubo)" \
       "python3 $QUBO_DIR/qubo_solvers/oriented_tangle/oriented_max_path.py mqlib $gfa_filepath $time_limit $outdir"

  bsub -J "$out_suffix.$seed.gurobi[1-$num_jobs]" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -q qpg -gpu - \
       -M "$memory" -o "$outdir/gurobi.$time_limit.%I.txt" -e "$outdir/error.gurobi.txt" -G "qpg"  \
       -w "done($out_suffix.$seed.build_qubo)" \
       "python3 $QUBO_DIR/qubo_solvers/oriented_tangle/oriented_max_path.py gurobi $gfa_filepath $time_limit $outdir"

  # bsub -J "$out_suffix.$seed.dwave[1-$num_jobs]" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -q qpg -gpu - \
  #      -M "$memory" -o "$outdir/dwave.$time_limit.%I.txt" -e "$outdir/error.dwave.txt" -G "qpg"  \
  #      -w "done($out_suffix.$seed.build_qubo)" \
  #      "python3 $QUBO_DIR/qubo_solvers/oriented_tangle/oriented_max_path.py dwave $gfa_filepath $time_limit $outdir"
done

# Tensor Networks
export COTENGRA_NUM_WORKERS=32
COTENGRA_DIR=/nfs/users/nfs_j/jc59/quantumwork/pangenome/cotengra_tensor_networks
source $COTENGRA_DIR/cotengra_venv/bin/activate

bsub -J "$out_suffix.$seed.cotengra" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -M "$memory"\
     -o "$outdir/cotengra.txt" -e "$outdir/error.cotengra.txt" -n 32\
     -w "done($out_suffix.$seed.build_qubo)" \
     -G "qpg" -q "qpg" -gpu - \
     "python3 $COTENGRA_DIR/non_local_exp.py $outdir $qubo_data_filepath"


# QOKit
QOKIT_DIR=/nfs/users/nfs_j/jc59/quantumwork/pangenome/qokit_simulation
source $QOKIT_DIR/qokit_venv/bin/activate
bsub -J "$out_suffix.$seed.qokit" -R '"select[mem>'$memory'] rusage[mem='$memory']"' -M "$memory"\
     -o "$outdir/qokit.txt" -e "$outdir/error.qokit.txt" -n 32\
     -w "done($out_suffix.$seed.build_qubo)" \
     -G "qpg" -q "qpg" -gpu - \
     "python3 $QOKIT_DIR/qokit_gpu.py $outdir $qubo_data_filepath"
exit 0