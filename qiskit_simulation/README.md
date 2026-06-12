# qiskit_simulation

Qiskit-based QAOA (Quantum Approximate Optimization Algorithm) simulation for pangenome tangle resolution. This module provides the full stack from Hamiltonian construction through circuit compilation, parameter optimisation, and result evaluation — for both standard QUBO and higher-order HUBO formulations.

## Overview

The `qiskit_qaoa` package contains four main submodules:

| Submodule | Description |
|-----------|-------------|
| `hubo/` | HUBO QAOA — higher-order Hamiltonian terms, SAT-based circuit compilation |
| `standard/` | Standard QAOA with Qiskit-Aer (statevector, MPS, GPU simulators) |
| `cvar/` | CVaR-QAOA and quantum advantage studies |
| `parameter_transfer/` | Classical MPS-based parameter pre-training |
| `utils/` | Shared infrastructure: Hamiltonians, transpiler passes, routing, sampling |

## Installation

```bash
pip install -e .
```

Dependencies: `qiskit`, `qiskit-aer`, `qiskit-ibm-runtime`, `scipy`, `scikit-optimize`, `gfapy`, `numpy`, `networkx`.

GPU simulation requires `qiskit-aer-gpu` and CUDA.

## Pipeline Overview

```
pangenome.gfa
    │
    ▼
gfa_utils.gfa_file_to_graph()          # GFA → oriented DiGraph
    │
    ├─[HUBO]─► hubo/graph_to_hubo_hamiltonian.py    # build SparsePauliOp
    │               │
    │          hubo/hubo_circuit_compilation.py      # SAT layout + gate routing
    │               │
    │          hubo/hubo_optimisation.py             # basin-hop / Bayesian optimise
    │
    └─[QUBO]─► utils/hamiltonian_utils.py           # Q matrix → Ising Hamiltonian
                    │
               utils/circuit_graph_utils.py         # QAOA ansatz construction
                    │
               standard/optimize_qaoa.py            # optimise parameters
                    │
               standard/sample_qaoa_circuit.py      # sample + score solution
```

## Submodule Reference

### `utils/` — Core Infrastructure

The utilities layer is used by all other submodules.

#### Hamiltonians & Evaluation

| File | Purpose |
|------|---------|
| `hamiltonian_utils.py` | Convert QUBO Q matrix to Ising Hamiltonian (`SparsePauliOp`); normalisation; decompose into interaction graph |
| `string_utils.py` | Energy evaluation from bitstrings; Pauli operator evaluation; print optimal solution statistics |
| `sample_utils.py` | Sample QAOA circuits, compute probability distributions over solutions |
| `gfa_utils.py` | Parse GFA files to orientation-aware DiGraphs (mirrors the HUBO module's version) |
| `postprocess.py` | Bitstring filtering using Gosper's hack to enumerate fixed-weight combinations |

#### Circuit Construction & Transpilation

| File | Purpose |
|------|---------|
| `qaoa_circuit_utils.py` | Uniform superposition initialisation, mixer operators |
| `circuit_graph_utils.py` | QAOA circuit construction from operator; circuit ↔ graph conversions |
| `qaoa_pass.py` | `QAOAPass`: Qiskit `TransformationPass` that inserts alternating cost/mixer layers into a circuit |
| `transpiler_passes.py` | `CommutingBlock`, `DecomposePauliZEvolution`, `FindCommutingPauliEvolutionsMulti` — group and decompose commuting Pauli evolutions |
| `pass_managers.py` | `get_hubo_pass_manager()`, `get_optimal_pass_manager()` — assemble preset pass-manager pipelines |
| `optimal_qaoa_pass_manager.py` | Additional preset pass-manager construction |

#### Routing & Layout

| File | Purpose |
|------|---------|
| `swap_strategy.py` | `ExtendedSwapStrategy` — swap layer management for line, grid, heavy-hex, and all-to-all topologies; distance tensor caching |
| `commuting_gate_router.py` | `CommutingGateRouter` — routes commuting Pauli evolution gates using a swap strategy |
| `commuting_gate_router_rzz.py` | RZZ-gate routing variant (lower depth for ZZ-dominated Hamiltonians) |
| `commuting_gate_router_precompute.py` | Pre-computes swap sequences for repeated compilation |
| `commuting_gate_router_precompute_rzz.py` | Precompute + RZZ combination (used by default in HUBO) |
| `commuting_gate_router_precompute_rzz_mask.py` | Masked variant for partial routing |
| `commuting_gate_router_all_to_all.py` | All-to-all topology specialisation (no swaps needed) |
| `commuting_gate_router_new.py` | Updated routing implementation |
| `routing_utils.py` | Greedy parity reduction, Gaussian elimination, CX network construction |
| `layout_utils.py` | Layout transformations between circuit representations |
| `sat_mapper.py` | `HigherOrderSatMapper` — SAT-based qubit layout for multi-qubit interactions; uses NuWLS solver |
| `backend_evaluator.py` | `BackendEvaluator` — selects best qubit subset from a real backend by fidelity |
| `implementable_interaction.py` | XOR / symmetric-difference subset finding (NP-hard placement subproblem) |
| `shortest_sequence_graph_reset.py` | Shortest-sequence graph utilities for circuit reset optimisation |

#### Optimisation & Utilities

| File | Purpose |
|------|---------|
| `qaoa_utils.py` | `bayesian_optimize_qaoa_parameters()`, `basinhopping_optimize_qaoa_parameters()`, `local_optimize_qaoa_parameters()` |
| `estimator_with_history.py` | `EstimatorWithHistory` — wraps Qiskit's `EstimatorV2`, recording all circuit evaluations during optimisation |
| `argparser.py` | Standard CLI argument parser used by all entry-point scripts |
| `logging.py` | Custom logger with separate stdout/stderr streams |

---

### `hubo/` — Higher-Order QUBO QAOA

HUBO QAOA using binary-encoded node indices (n = ⌈log₂(V)⌉ qubits per timestep).

| File | Purpose |
|------|---------|
| `graph_to_hubo_hamiltonian.py` | GFA graph → HUBO `SparsePauliOp` with higher-order constraint penalties |
| `hubo_circuit_compilation.py` | Full compilation pipeline: SAT layout + `CommutingGateRouterPrecomputeRzz` |
| `hubo_circuit_compilation_for_simulation.py` | Simulation-specific variant (no hardware transpilation) |
| `hubo_optimisation.py` | Main HUBO parameter optimisation CLI (basin-hopping, MPS backend) |
| `hubo_optimisation_default.py` | Default noise-model variant |
| `hubo_optimisation_per_layer.py` | Per-layer parameter optimisation |
| `hubo_optimisation_no_noise_all_to_all.py` | No-noise all-to-all simulation |
| `hubo_optimisation_no_noise_all_to_all_single.py` | Single-shot variant |
| `hubo_optimisation_no_noise_all_to_all_sweep.py` | Parameter sweep variant |
| `hubo_optimisation_for_simulation.py` | Uses pre-compiled circuit from `_for_simulation` |
| `get_circuit_depths.py` | Circuit depth analysis across test instances |
| `get_circuit_depths_precompute.py` | Precomputed routing variant of depth analysis |
| `plot_hubo.py`, `plot_hubo_for_simulation.py`, `plot_per_layer_hubo.py` | Visualisation |
| `profile_precompute_router.py` | Routing performance profiling |

**Typical usage:**

```bash
# Compile HUBO circuit
python qiskit_qaoa/hubo/hubo_circuit_compilation.py \
    -f <filename_id> -c <copy_numbers> -C grid -t 60

# Optimise QAOA parameters
python qiskit_qaoa/hubo/hubo_optimisation.py \
    -f <filename_id> -c <copy_numbers> -p 3 -n 1024
```

---

### `standard/` — Standard QAOA

General QAOA with QUBO/Ising cost Hamiltonians and Qiskit-Aer simulators.

| File | Purpose |
|------|---------|
| `optimize_qaoa.py` | Main optimisation CLI; supports GPU, MPS, statevector; basin-hopping / Bayesian / local |
| `sample_qaoa_circuit.py` | Load optimised parameters, sample circuit, score solution |
| `multi_level_experiment.py` | Multi-level QAOA: optimise p=1, extrapolate to p=2,3,… using DCT/DST |
| `multilevel_with_estimator.py` | Multi-level variant using `EstimatorWithHistory` |
| `mps_optimize_qaoa.py` | MPS-specific optimisation (matrix product state simulator) |
| `dump_qaoa_circuit.py` | Export optimised QAOA circuit to QASM format |
| `cacheblocking.py` | `AerSimulator` cache-blocking memory utilities for large simulations |
| `test.py` | GPU vs cuStateVec benchmarking |

**Typical usage:**

```bash
python qiskit_qaoa/standard/optimize_qaoa.py \
    -f data/hla_drb1/hla.gfa -p 3 --method basinhopping --shots 1024

python qiskit_qaoa/standard/sample_qaoa_circuit.py \
    -f data/hla_drb1/hla.gfa -p 3
```

---

### `cvar/` — CVaR QAOA

Conditional Value-at-Risk (CVaR) variant which optimises the expected energy of the bottom-α fraction of samples rather than the full expectation value. This reduces the influence of high-energy (bad) samples and has been linked to quantum advantage in certain regimes.

| File | Purpose |
|------|---------|
| `optimize.py` | CVaR parameter optimisation (AER or hardware backend) |
| `optimize_sweep.py` | Sweep over α values |
| `optimize_no_shot_noise.py` | Shot-noise-free simulation |
| `optimize_no_shot_noise_sweep.py` | Shot-noise-free sweep |
| `optimize_hardware.py`, `optimize_hardware_bias.py` | Real IBM hardware variants |
| `optimize2.py` | Updated optimisation variant |
| `hubo_qaoa.py` | CVaR-HUBO combination |
| `sample_hardware.py` | Sample from hardware-optimised circuit |
| `qaoa_depth_width.py` | Circuit depth/width analysis for CVaR circuits |
| `plot_cvar.py`, `plot_cvar2.py`, `plot_hardware.py`, `plot_hardware_sample.py`, `paper_plot.py` | Visualisation |

---

### `parameter_transfer/`

| File | Purpose |
|------|---------|
| `classical_training.py` | Train QAOA parameters classically using MPS simulation, then transfer to quantum hardware |

---

## Shell Script Wrappers (`bin/`)

Scripts in `bin/` provide bsub job submission wrappers for the main Python entry points, handling HPC arguments (memory, GPU allocation, job arrays).

## Output Paths

Results are written to Lustre scratch: `/lustre/scratch127/qpg/jc59/`.
