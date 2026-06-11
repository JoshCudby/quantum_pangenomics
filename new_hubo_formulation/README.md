# new_hubo_formulation

HUBO (Higher-Order Unconstrained Binary Optimization) formulation for tangle resolution. Instead of one binary variable per (timestep, node, orientation) triple as in the QUBO formulation, this module encodes the identity of the node visited at each timestep as a binary integer — dramatically reducing the number of qubits at the cost of higher-order (3-body, 4-body, …) Pauli interaction terms.

## Key Difference from the QUBO Formulation

| Property | QUBO (`new_qubo_formulation`) | HUBO (`new_hubo_formulation`) |
|----------|-------------------------------|-------------------------------|
| Variables per timestep | V · 2 (one per node/orientation) | n = ⌈log₂(V·2)⌉ (binary encoding) |
| Total circuit width | T · V · 2 | T · n |
| Constraint order | Quadratic (2-body) | Higher-order (3–10+ body) |
| Main challenge | Large qubit count for long paths | Compiling high-order terms to 2-qubit hardware gates |
| Circuit approach | Built on-the-fly | Pre-compiled offline via SAT mapping |

For a graph with V=16 nodes and T=8 timesteps: QUBO uses 8×16×2=256 qubits; HUBO uses 8×⌈log₂(32)⌉=8×5=40 qubits.

## Directory Structure

```
new_hubo_formulation/
├── hubo_qaoa/
│   ├── utils/                         # Core HUBO utilities
│   │   ├── gfa_utils.py               # GFA → oriented DiGraph + encoding parameters
│   │   ├── graph_to_hubo_hamiltonian.py  # HUBO Hamiltonian construction
│   │   ├── get_swap_strategy.py       # Topology-matched swap strategy selection
│   │   ├── parameterise_circuit.py    # Re-parameterise compiled circuit for optimisation
│   │   ├── lr_qaoa.py                 # Linear-ramp QAOA (uses pre-compiled circuits)
│   │   ├── iterative_qaoa_utils.py    # Boltzmann warm-start (no fixed-weight postprocessing)
│   │   └── str_utils.py              # Binary string enumeration (small instances)
│   ├── circuit_depths.py              # Sweep swap layers to minimise 2-qubit gate depth
│   ├── compilation.py                 # Full compilation pipeline with term-subset selection
│   ├── get_circuit_depths_all_to_all_new.py  # Depth analysis for all-to-all topology
│   ├── nonvariational/                # Simulation experiments
│   │   ├── nonvariational.py          # Iterative warm-start QAOA (uses compiled circuit)
│   │   ├── param_exploration.py       # Parameter landscape sweep
│   │   ├── performance_diagram.py
│   │   ├── nonvariational_hardware.py # Real hardware variant
│   │   └── plotting/                  # Figure generation scripts
│   └── notebooks/                     # Jupyter analysis notebooks
├── ionq/                              # IonQ-specific experiments
└── bin/                              # Compiled binaries
```

## Two-Phase Workflow

The HUBO pipeline is split into an offline **compilation** phase and an online **simulation** phase.

### Phase 1 — Compilation (offline, compute-intensive)

```bash
python hubo_qaoa/compilation.py \
    -f <filename_id> \
    -c <copy_numbers> \
    -C grid \                # topology: grid / line / heavy-hex / all
    -t 60 \                  # SAT solver timeout (seconds)
    --times-to-keep 0 3 7 \  # which timestep constraint terms to include
    --fraction-four 0.5 \    # relative weight of 4-body terms
    --fraction-six 0.2       # relative weight of 6-body terms
```

This:
1. Builds the full HUBO Hamiltonian (`graph_to_hubo_hamiltonian.py`)
2. Selects a subset of constraint terms (controlled by `--times-to-keep` and fraction flags)
3. Initialises the SAT mapper and swap strategy for the chosen topology
4. Sweeps over swap layer counts to find the circuit with minimum 2-qubit gate depth
5. Saves a pickle: `{full_hamiltonian, compiled_hamiltonian, best_circuit}`

### Phase 2 — Simulation (online, uses compiled circuit)

```bash
python hubo_qaoa/nonvariational/nonvariational.py \
    -f <filename_id> \
    -c <copy_numbers> \
    -n 1024              # measurement shots
```

This:
1. Loads the compiled circuit from the Phase 1 pickle
2. Re-parameterises the circuit with a new γ parameter (`parameterise_circuit.py`)
3. Applies the linear-ramp schedule (Δβ=0.75, Δγ=0.30)
4. Runs iterative Boltzmann warm-start (5 iterations)
5. Saves energies, samples, and the circuit to a pickle

## Key Utilities

### `utils/gfa_utils.py` — GFA Parsing

`gfa_file_to_graph(filepath, copy_numbers)` reads a GFA file and returns:
- `graph`: orientation-aware DiGraph (each original segment → two nodes `i+`, `i−`)
- `n`: number of encoding bits, `n = ⌈log₂(V)⌉`
- `V`: total number of oriented nodes (2 × number of segments)
- `total_weight`: sum of copy numbers / 2 (used for Hamiltonian normalisation)

### `utils/graph_to_hubo_hamiltonian.py` — Hamiltonian Construction

`graph_to_hubo_hamiltonian(graph, n, T, lamda, constraint_terms)` builds a `SparsePauliOp` encoding:

- **Objective**: maximise coverage weighted by node copy numbers (2-body Z terms)
- **Constraint terms**: for each selected timestep t, enforce that the node active at t and the node active at t+1 are connected by a graph edge; encoded as multi-qubit Pauli products over the n-bit registers for t and t+1

`constraint_terms` can be a float (fraction of timesteps) or a tuple of explicit timestep indices. Only a subset of timesteps is enforced in practice to keep circuit depth manageable; the `--times-to-keep` flag in `compilation.py` controls this.

Returns `(normalised_hamiltonian, normalisation_factor)`.

### `utils/get_swap_strategy.py` — Topology Selection

`get_swap_strategy(coupling_map, n, T)` selects a swap strategy matching the given coupling map:

| Coupling map | Strategy |
|---|---|
| `'line'` | Linear chain of n·T qubits |
| `'grid'` | 2D grid, rows=⌈√(n·T)⌉ |
| `'all'` | All-to-all (no swaps needed) |
| `'heavy-hex'` | IBM heavy-hex lattice, auto-sized to contain n·T qubits |

### `utils/parameterise_circuit.py` — Re-parameterisation

`parameterise_circuit(qc, parameter)` takes a compiled circuit (which has fixed evolution times set during compilation) and replaces each gate's time parameter with `time * parameter`. This exposes a single scalar γ that can be swept during optimisation, rather than running the full compilation again.

### `utils/lr_qaoa.py` — Linear-Ramp Schedule

Same linear-ramp approach as the QUBO module, but operates on pre-compiled cost circuits. The default HUBO parameters are Δβ=0.75, Δγ=0.30.

### `utils/iterative_qaoa_utils.py` — Boltzmann Warm-Start

Identical algorithm to the QUBO version but without the fixed-weight post-processing step (the HUBO encoding does not have a simple fixed-weight constraint). Uses 5 iterations by default (fewer than QUBO due to higher per-circuit cost).

## Circuit Compilation Details

`circuit_depths.py` implements `sweep_swap_depths()` which:
1. Tests multiple numbers of swap layers (each layer rearranges qubit connectivity)
2. For each layer count, routes the Hamiltonian terms using `CommutingGateRouterPrecomputeRzz` — groups commuting Pauli evolutions and finds an efficient gate ordering
3. Applies `HigherOrderSatMapper` (SAT-based qubit layout) when `timeout > 0`
4. Records gate depth and 2-qubit gate count for each configuration
5. Fine-searches around the best configuration

The key optimisation passes used:
- `FindCommutingPauliEvolutionsMulti` — groups commuting evolution gates
- `CommutingGateRouterPrecomputeRzz` — routes with precomputed RZZ gate library
- Inverse cancellation and commutative cancellation — reduce gate count

## Term Selection Trade-off

The `--fraction-four` and `--fraction-six` flags in `compilation.py` control what fraction of 4-body and 6-body Pauli terms are included relative to 2-body terms. Including more high-order terms improves constraint enforcement but increases circuit depth. The optimal trade-off depends on the problem size and available coherence time.
