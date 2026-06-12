# jc-tangle

A research platform for solving **pangenome tangle resolution** using quantum and classical optimization. Given a sequence assembly graph in GFA format, the goal is to find the highest-coverage path through ambiguous regions (tangles) by formulating the problem as a QUBO or HUBO and solving it with multiple backends.

## Background

Pangenome assembly graphs produced by tools like [syncasm](https://github.com/c-zhou/oatk) contain regions where multiple haplotypes overlap ambiguously. Resolving these tangles — finding the path that maximises sequence coverage and minimises breaks — can be cast as a combinatorial optimisation problem. This repository frames that problem as a Quadratic (QUBO) or Higher-Order (HUBO) Unconstrained Binary Optimisation and benchmarks several solvers against each other.

## Conceptual Workflow

```
GFA file (assembly graph)
    │
    ▼
build_*_qubo_matrix.py  ──── encode path problem as QUBO/HUBO matrix
    │
    ├──► D-Wave quantum annealer   (dwave-ocean-sdk)
    ├──► MQLib classical QUBO solver
    ├──► Gurobi classical MILP solver - used here only for QUBO problems
    └──► QAOA simulation or experiments (Qiskit)
    │
    ▼
path solution  ──── scored by coverage, breaks, identity
```

## Repository Structure

| Directory | Description |
|-----------|-------------|
| `bin/` | HPC benchmark orchestration: shell scripts for job submission, sequencing simulation, stat aggregation |
| `qubo_solvers/` | Core installable Python package — QUBO matrix builders and solver wrappers (D-Wave, MQLib, Gurobi) |
| `new_qubo_formulation/` | QUBO formulations for QAOA circuits, with non-variational parameter strategies |
| `new_hubo_formulation/` | HUBO formulation using binary-encoded node indices; circuit compilation and simulation |
| `qiskit_simulation/` | Qiskit-based QAOA simulation: standard QUBO, HUBO, CVaR variants, circuit compilation and parameter optimisation |
| `pytket_simulation/` | PyTket-based QAOA simulation (legacy)|
| `sat/` | SAT solver approaches |
| `data/` | Test GFA datasets (Arabidopsis, Daphnia, HLA, PhiX174, synthetic) |
| `utils/` | Shared utilities (GFA subgraph cropper, plotting notebooks) |
| `modules/` | Git submodules (MQLib, QOKit, qpg, oatk, etc.) |
|-----------|-------------|
| `qokit_simulation/` | QOKit tensor-network simulation (legacy)|
| `openqaoa_simulation/` | OpenQAOA framework integration (legacy)|
| `cotengra_tensor_networks/` | Tensor network contraction experiments (legacy)|
| `prog_qaoa/` | Programmable QAOA parameter experiments (legacy)|
| `pathfinder/` | Classical exhaustive path finder (baseline) |

## Installation

#### TODO: check this works!
Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/JoshCudby/pangenome.git
cd jc-tangle
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

Install the main solver package:

```bash
pip install -e qubo_solvers/
```

Core Python dependencies:

```
numpy
networkx
gfapy          # GFA file parsing
dwave-ocean-sdk # D-Wave quantum annealing (requires API key)
gurobipy        # Gurobi solver (requires licence)
qiskit          # Quantum circuit simulation
```


### Solver credentials

**Gurobi**: obtain a licence at https://support.gurobi.com/hc/en-us/articles/14799677517585

**D-Wave**: create an account at https://cloud.dwavesys.com/leap/, then configure the CLI:

```bash
dwave config create
```

## Quick Start

Build a QUBO matrix from a GFA file and solve with MQLib:

```bash
# 1. Build the QUBO matrix (saves a pickle to DATA_DIR)
python qubo_solvers/qubo_solvers/tangle/build_tangle_qubo_matrix.py \
    -f data/hla_drb1/hla.gfa \
    -d /path/to/output

# 2. Solve with MQLib at time limits 5, 30, 120 seconds
python qubo_solvers/qubo_solvers/tangle/max_path.py \
    -f data/hla_drb1/hla.gfa \
    -d /path/to/output \
    --solver mqlib \
    --time-limits 5 30 120
```

For a complete benchmark across all solvers and multiple seeds, see `bin/full_benchmark_tangle.sh`.

## Data Format

Input: GFA (Graph Fragment Assembly) files. Nodes should carry an `SC` tag for coverage, e.g.:

```
S  node1  ACGT...  SC:f:42.0
L  node1  +  node2  +  *
```

Output: pickle files containing the solution path, coverage scores, and timing information, written to a Lustre scratch directory (`/lustre/scratch127/qpg/jc59/`).

## HPC Notes

Benchmarks run on an LSF cluster via `bsub`. Scripts in `bin/` use GNU parallel (128 jobs) and submit chains of dependent jobs. Key paths are hardcoded in `qubo_solvers/qubo_solvers/definitions.py`:

```python
DATA_DIR = '/lustre/scratch127/qpg/jc59/data'
OUT_DIR  = '/lustre/scratch127/qpg/jc59/out'
```

Adjust these before running on a different system.

## Directory READMEs

Each major module has its own README with usage details:

- [`qubo_solvers/README.md`](qubo_solvers/README.md)
- [`new_qubo_formulation/README.md`](new_qubo_formulation/README.md)
- [`new_hubo_formulation/README.md`](new_hubo_formulation/README.md)
- [`bin/README.md`](bin/README.md)
