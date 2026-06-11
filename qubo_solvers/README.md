# qubo_solvers

Installable Python package (`pip install -e .`) implementing QUBO matrix builders and solver wrappers for pangenome tangle resolution. This is the primary package used in the HPC benchmarks.

## Installation

```bash
pip install -e qubo_solvers/
# or from within this directory:
pip install -e .
```

Dependencies: `gfapy`, `gurobipy`, `dwave-ocean-sdk`, `numpy`, `networkx`

## Package Structure

```
qubo_solvers/
├── definitions.py           # Core enums (Solver) and dataclasses (QuboDescription)
├── logging.py               # Shared logging configuration
├── pathfinder_coverage.py   # Calls the Pathfinder binary to extract node copy numbers
├── tangle/                  # Standard (unoriented) tangle QUBO
├── oriented_tangle/         # Orientation-aware QUBO (edges carry orientation)
├── edge_tangle/             # Edge-variable QUBO formulation
└── diploid_tangle/          # Diploid / polyploid variant
```

## Shared Definitions (`definitions.py`)

**`Solver` enum**: selects the backend:

| Value | Backend |
|-------|---------|
| `Solver.DWAVE` | D-Wave Leap hybrid quantum-classical solver |
| `Solver.MQLIB` | MQLib classical QUBO heuristic solver |
| `Solver.GUROBI` | Gurobi commercial LP/MILP solver |

**`QuboDescription` dataclass**: bundles all inputs needed to run a solve:

| Field | Type | Description |
|-------|------|-------------|
| `filename` | `str` | Path to input GFA file |
| `data_dir` | `str` | Directory for output pickle files |
| `graph` | `Graph` | NetworkX graph with copy-number annotations |
| `time_limits` | `list[int]` | Solver wall-clock time limits in seconds |
| `jobs` | `int` | Number of parallel solver runs |
| `Q` | `ndarray` | QUBO matrix |
| `offset` | `int` | Constant energy offset from constraint terms |
| `T` | `int` | Maximum path length (number of timesteps) |
| `V` | `int` | Number of original graph nodes |
| `solver` | `Solver` | Which backend to use |

## Workflow

Each submodule follows the same two-step pattern:

### Step 1 — Build the QUBO matrix

```bash
python qubo_solvers/tangle/build_tangle_qubo_matrix.py \
    -f <path/to/graph.gfa> \
    -d <output_directory>
```

This reads the GFA, runs Pathfinder to get per-node copy numbers, constructs the QUBO matrix Q, and saves a pickle containing `{Q, offset, T_max, V, graph}`.

### Step 2 — Solve

```bash
python qubo_solvers/tangle/max_path.py \
    -f <path/to/graph.gfa> \
    -d <output_directory> \
    --solver mqlib \
    --time-limits 5 30 120 \
    --jobs 4
```

This loads the pickle, dispatches to the chosen solver at each time limit, validates the resulting path, and writes solution pickles.

## Submodules

### `tangle/` — Standard Tangle QUBO

Unoriented formulation. Each binary variable `x[t, i]` indicates whether node `i` is visited at timestep `t`. Penalty terms enforce:
- **One-hot per timestep**: exactly one node per time step
- **Edge validity**: consecutive nodes must be connected in the graph
- **Coverage objective**: reward proportional to node copy numbers

Key files:
- `build_tangle_qubo_matrix.py` — CLI entry point; calls `graph_with_copy_numbers()` then `get_tangle_qubo_matrix()`
- `max_path.py` — solver dispatcher; calls `dwave_sample_qubo()`, `mqlib_sample_qubo()`, or `gurobi_sample_qubo()` then `validate_path()`
- `max_path_timing.py` — timing-instrumented variant of `max_path.py`
- `utils/graph_utils.py` — `graph_with_copy_numbers()`: annotates NetworkX graph nodes with copy numbers from Pathfinder
- `utils/qubo_utils.py` — `get_tangle_qubo_matrix()`: assembles the Q matrix from constraint and objective terms
- `utils/sampling_utils.py` — solver integration functions and path validation

### `oriented_tangle/` — Orientation-Aware QUBO

Variables `x[t, i, σ]` where `σ ∈ {+, −}` encode both which node and which strand is visited. This doubles the variable count but captures strand-specific assembly constraints.

Key files:
- `build_oriented_qubo_matrix.py` — builds oriented QUBO; also exports MQLib-format matrix
- `build_edge2node_qubo_matrix.py` — converts an edge-based QUBO to a node-based one
- `oriented_max_path.py` — solver dispatcher for the oriented formulation
- `utils/graph_utils.py`, `utils/qubo_utils.py` — oriented variants of the standard utilities

### `edge_tangle/` — Edge-Variable Formulation

Variables represent edges rather than nodes, giving a more direct encoding of path continuity. Includes D-Wave Constrained Quadratic Model (CQM) support.

Key files:
- `build_edge_qubo_matrix.py` — builds edge-based QUBO
- `edge_cqm_dwave.py` — D-Wave CQM formulation
- `edge_max_path_dwave.py`, `edge_max_path_gurobi.py` — solver dispatchers
- `edge_known_mqlib.py` — MQLib with a known-solution warm start

### `diploid_tangle/` — Diploid / Polyploid Variant

Extends the tangle formulation to handle diploid or polyploid assembly graphs where copy numbers are expected to be ≥2.

## Compiled Binaries (`bin/`)

The `bin/` subdirectory contains compiled C++ solver executables and compilation scripts:

- `compile_all_tangle.sh`, `compile_all_oriented_tangle.sh`, etc. — compile the MQLib-based binaries for each formulation
- `tangle/`, `oriented_tangle/`, `edge_tangle/`, `diploid_tangle/` — compiled executables
- `path2seq.pl` — Perl script that converts a solution path (list of node IDs) to the corresponding nucleotide sequence

## Output Paths

By default, pickles are written to Lustre scratch:

```
DATA_DIR = '/lustre/scratch127/qpg/jc59/data'
OUT_DIR  = '/lustre/scratch127/qpg/jc59/out'
```

Override these in `definitions.py` for local development.
