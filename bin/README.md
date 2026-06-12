# bin/

## Legacy

Most of these scripts are superseded by the overall pipeline from the QPG or schizophrenia repos.
Each directory contains /bin subdirectories containing relevant orchestration scripts for that directory.

## Summary

HPC benchmark orchestration scripts for the tangle resolution pipeline. Scripts use LSF (`bsub`) for job submission and GNU parallel for local parallelism. They are written in Bash and Perl and are designed to run on the Sanger Institute compute cluster with Lustre scratch storage.

## Typical Pipeline

```
rotate_sequence.pl          # rotate input sequence to generate variants
    │
    ▼
sequencing simulation       # shred to short reads (20 kb, 0.1% error, 30x coverage)
    │
    ▼
syncasm                     # assemble reads into pangenome graph (.gfa)
    │
    ▼
build_tangle_qubo_matrix.py # build QUBO matrix from GFA + copy numbers
    │
    ├──► Pathfinder (exhaustive classical baseline)
    ├──► MQLib (classical QUBO, multiple time limits)
    ├──► Gurobi (commercial LP/MILP)
    └──► D-Wave (quantum annealer)
    │
    ▼
stat aggregation scripts    # aggregate coverage/break metrics across seeds and time limits
```

## Script Reference

### Benchmark Execution

| Script | Description |
|--------|-------------|
| `full_benchmark_tangle.sh` | Runs 5 seeds of the complete tangle benchmark: rotates sequences, simulates reads, assembles, builds QUBO, submits solver jobs with bsub dependencies |
| `full_benchmark_oriented.sh` | Same as above but uses the orientation-aware QUBO formulation (`oriented_tangle`) |
| `run_full_benchmark_experiment.sh` | Master experiment runner: sweeps annotate modes (ga/km/mg) and time limits using GNU parallel (128 jobs), 5 seeds |
| `tangle_resolution_benchmark.sh` | Low-level benchmark runner using GNU parallel with a config file |
| `tangle_resolution_benchmark_max.sh` | Extracts best results from benchmark summary files, grouping by batch/table/row |

### Statistics & Aggregation

| Script | Description |
|--------|-------------|
| `average_gfa_sim.sh` | Computes column means from GFA simulation output tables (len-t, len-q, covered, ncontig, nbreaks, nindel, identity) |
| `max_gfa_sim.sh` | Finds the best-performing run across batches (maximises average coverage across tables) |
| `print_avg_coverage.sh` | Extracts and averages coverage metrics per solver (mqlib/dwave/gurobi/pathfinder) and time limit |
| `get_gfa_run_stats.sh` | Aggregates statistics across 50 seeds for a single solver at time limits 3, 5, 15, 30, 60, 120 s |
| `get_gfa_seed_stats.sh` | Single-seed statistics extractor (mode 2, seed 6 — useful for quick debugging) |
| `tangle_resolution_benchmark_stats.sh` | Batch processor for tangle benchmarks: filters by data version, finds max coverage per row |
| `find_min_average_coverage.sh` | Finds the table with the minimum average coverage across a set of result files |

### Utilities

| Script | Description |
|--------|-------------|
| `rotate_sequence.pl` | Perl script that rotates each FASTA sequence by a random position, generating sequence variants for benchmarking |
| `bubble_finder.sh` | Finds bubble structures in assembly graphs using `pangene.js`, filters by `min_size`, and extracts subgraphs with `gfa_cropper.py` |
| `copy_gfa.sh` | Matches S (sequence) and L (link) lines between two GFA files; outputs matching rows from the second file |
| `node_avg.sh` | Node averaging utility (minimal / deprecated) |
| `test.sh` | Test harness |

### Reference Files

| File | Description |
|------|-------------|
| `full_benchmark_experiment.md` | Human-readable description of the full benchmark pipeline |
| `run_sim_command.txt` | Archive of example bsub commands for various solver/seed combinations |

## Environment Requirements

- **LSF / bsub**: job scheduler (Sanger cluster)
- **GNU parallel**: `parallel` must be on `PATH`
- **syncasm**: genome assembler from [oatk](https://github.com/c-zhou/oatk) (`modules/oatk`)
- **pangene.js**: pangenome utilities (available via `modules/qpg`)
- Lustre scratch storage mounted at `/lustre/scratch127/qpg/jc59/`
