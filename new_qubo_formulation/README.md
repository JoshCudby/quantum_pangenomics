# new_qubo_formulation

QUBO formulations and QAOA circuit utilities for the tangle resolution problem. This module builds on the core `qubo_solvers` package and focuses on quantum simulation, particularly **non-variational QAOA** using a linear-ramp parameter schedule with warm-start refinement.

## Problem Encoding

The tangle path is encoded as a set of time-indexed binary variables:

```
x[t, i, σ]  =  1  if node i with orientation σ ∈ {+, −} is visited at timestep t
```

The QUBO matrix Q has shape `(T · V · 2) × (T · V · 2)`, where T is the maximum path length and V is the number of original graph nodes. Three penalty terms are added to enforce valid paths:

| Term | Penalty | Meaning |
|------|---------|---------|
| One-hot per timestep | λ_t | Exactly one (node, orientation) is active at each t |
| Edge validity | λ_g | Consecutive active nodes must be connected by an edge |
| Boundary | λ_g | Start (t=0) and end (t=T−1) nodes must also obey edge constraints |
| Coverage objective | λ_w | Reward active nodes proportional to their copy numbers (squared) |

## Directory Structure

```
new_qubo_formulation/
├── qubo_qaoa/
│   ├── oriented_tangle/          # Builds QUBO matrix from GFA
│   │   ├── build_oriented_qubo_matrix.py
│   │   └── utils/
│   │       ├── graph_utils.py    # GFA → oriented DiGraph
│   │       └── qubo_utils.py     # QUBO matrix construction
│   ├── utils/                    # Shared QAOA circuit utilities
│   │   ├── lr_qaoa.py            # Linear-ramp QAOA parameter schedule
│   │   ├── circuit_construction.py  # Full QAOA circuit assembly
│   │   ├── swap_strategy.py      # Qubit topology routing
│   │   ├── iterative_qaoa_utils.py  # Boltzmann warm-start refinement
│   │   ├── postprocess.py        # Fixed-weight constraint enforcement
│   │   └── str_utils.py          # Binary string enumeration (small instances)
│   ├── nonvariational/           # Simulation experiments
│   │   ├── nonvariational.py     # Warm-start iterative QAOA run
│   │   ├── param_exploration.py  # Parameter landscape sweep (Δβ, Δγ, p)
│   │   ├── performance_diagram.py
│   │   ├── nonvariational_hardware.py  # Real hardware variant
│   │   ├── get_circuit_depth_widths.py
│   │   ├── phylogeny/            # Phylogenetic tree problem variant
│   │   └── plotting/             # Publication-quality figure scripts
│   └── test/                     # Unit tests
└── bin/                          # Compiled binaries (same role as qubo_solvers/bin)
```

## Workflow

### 1. Build the QUBO matrix

```bash
python qubo_qaoa/oriented_tangle/build_oriented_qubo_matrix.py \
    -f <path/to/graph.gfa> \
    -c <copy_numbers>            # space-separated per-node copy numbers
    -p <lambda_t> <lambda_g> <lambda_w>   # penalty weights
    -d <output_directory>
```

Output: pickle containing `{Q, offset, T_max, V, graph}` and an MQLib-format text matrix.

### 2. Explore the parameter space

```bash
python qubo_qaoa/nonvariational/param_exploration.py \
    -f <filename_id> \
    -n 1024          # measurement shots
```

Sweeps over `p` (circuit depth), `Δβ`, and `Δγ` and saves a grid of mean energies and sample collections to a pickle.

### 3. Run a warm-start iterative QAOA simulation

```bash
python qubo_qaoa/nonvariational/nonvariational.py \
    -f <filename_id> \
    -N <num_nodes>   # sets initial single-qubit rotation angle ≈ 1/(2N)
    -T <runtime>
    -n 1024
```

Runs iterative QAOA with fixed linear-ramp parameters (Δβ=0.63, Δγ=0.16) and a Boltzmann temperature schedule.

### 4. Plot results

Scripts in `nonvariational/plotting/` convert output pickles to figures.

## Key Utilities

### `utils/lr_qaoa.py` — Linear-Ramp Schedule

The linear-ramp parameterisation avoids variational optimisation by using a fixed parameter schedule:

```
β_j = Δβ · (1 − (j − 0.5) / p)
γ_j = Δγ · (j − 0.5) / p
```

where j = 1…p is the layer index. `get_LR_qaoa_circuit()` assembles the full p-layer QAOA circuit. `get_hardware_LR_qaoa_circuit()` additionally transpiles for IBM-compatible backends.

### `utils/circuit_construction.py` — Circuit Assembly

`circuit_construction()` builds a complete QAOA circuit by:
1. Separating Pauli evolution terms into single-qubit (Z) and two-qubit (ZZ) gates
2. Routing two-qubit gates using a `SwapStrategy` matched to the target topology
3. Applying alternating even/odd mixer layers
4. Mapping physical qubits back to logical measurements

### `utils/swap_strategy.py` — Qubit Routing

`QUBOSwapStrategy` extends Qiskit's `SwapStrategy` with QUBO-specific constructors:

| Method | Topology |
|--------|----------|
| `from_all_to_all(n)` | Fully connected |
| `from_line(line, layers)` | Linear chain |
| `from_grid(rows, cols)` | 2D grid with checkerboard swaps |
| `from_heavy_hex(rows, cols)` | IBM heavy-hex lattice |

### `utils/iterative_qaoa_utils.py` — Boltzmann Warm-Start

Implements iterative parameter refinement:

1. Run QAOA circuit, measure bitstring samples
2. Evaluate energy of each sample under the cost Hamiltonian
3. Compute Boltzmann-weighted bias per qubit: `bias_q = Σ_s P(s) · s_q`
4. Convert biases to rotation angles: `θ_q = 2 · arcsin(√bias_q)`
5. Use angles to warm-start the next iteration (RY rotations on |+⟩ state)
6. Anneal inverse temperature β_T upward each iteration

Hyperparameters: `eta=1`, `eps=0.05`, `alpha=1.0`, `max_beta_T=0.15`.

### `utils/postprocess.py` — Fixed-Weight Enforcement

After sampling, `postprocess()` applies `sample_fixed_weight()` to each bitstring: if a sample has more than T ones, it uniformly selects exactly T of them. This enforces the cardinality constraint from the path formulation without adding a hard constraint to the circuit.

### `utils/str_utils.py`

`genbin(n)` yields all 2^n binary strings of length n — used for exact energy evaluation on small (≤20 qubit) instances.
