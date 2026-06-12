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
| `new_qubo_formulation/` | (Quantum only) QUBO formulations for QAOA circuits, with non-variational parameter strategies |
| `new_hubo_formulation/` | (Quantum only) HUBO formulation using binary-encoded node indices; circuit compilation and simulation |
| `qiskit_simulation/` | Qiskit-based QAOA simulation: standard QUBO, HUBO, CVaR variants, circuit compilation and parameter optimisation |
| `sat/` | SAT solver approaches |
| `data/` | Test GFA datasets (Arabidopsis, Daphnia, HLA, PhiX174, synthetic) |
| `utils/` | Shared utilities (GFA subgraph cropper, plotting notebooks) |
| `legacy_code/` | Legacy implementations (QOKit, OpenQAOA, tensor networks, programmable QAOA, pathfinder) |

## Installation

### Prerequisites

- Python ≥ 3.10
- [MQLib](https://github.com/MQLib/MQLib) — build from source following the instructions in that repo and add the resulting binary to your `$PATH`

### Quick install

```bash
git clone https://github.com/JoshCudby/jc-tangle.git
cd jc-tangle
bash install.sh
```

`install.sh` walks through every step below and prints reminders about the two manual configuration steps at the end.

### Step-by-step

**1. Clone**

```bash
git clone https://github.com/JoshCudby/jc-tangle.git
cd jc-tangle
```

**2. Create a virtual environment (recommended)**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install MQLib**

Build MQLib from [github.com/MQLib/MQLib](https://github.com/MQLib/MQLib) and ensure the binary is on your `$PATH`. MQLib is a C++ project with its own build instructions.

**4. Install Python packages**

All four packages use `pip install -e` (editable install via [flit](https://flit.pypa.io/)):

```bash
pip install -e qubo_solvers/
pip install -e new_qubo_formulation/
pip install -e new_hubo_formulation/
pip install -e qiskit_simulation/
```

Install only the packages relevant to your workflow — see the table below.

**5. Set local output paths**

Edit `qubo_solvers/qubo_solvers/definitions.py` and update `DATA_DIR` and `OUT_DIR` to directories that exist on your machine. The defaults point to Sanger Lustre scratch and will not work elsewhere.

**6. Configure solver credentials**

**Gurobi**: obtain a licence at https://support.gurobi.com/hc/en-us/articles/14799677517585

**D-Wave**: create an account at https://cloud.dwavesys.com/leap/, then:

```bash
dwave config create
```

### Package overview

| Package | Key dependencies | What it enables |
|---|---|---|
| `qubo_solvers/` | `gfapy`, `gurobipy`, `dwave-ocean-sdk`, `numpy`, `networkx` | QUBO matrix builders; D-Wave, MQLib, and Gurobi solver wrappers — the core HPC benchmark package |
| `new_qubo_formulation/` | `qiskit`, `qiskit-optimization`, `qiskit-ibm-runtime`, `scipy`, `scikit-optimize` | QUBO formulations for QAOA circuits with non-variational parameter strategies |
| `new_hubo_formulation/` | same as above | HUBO formulation using binary-encoded node indices; circuit compilation and simulation |
| `qiskit_simulation/` | same + `qiskit-aer-gpu` | Full Qiskit QAOA simulation stack: standard QUBO, HUBO, CVaR variants, GPU-accelerated simulation |

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
