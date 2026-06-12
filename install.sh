#!/usr/bin/env bash
set -euo pipefail

# ── Prerequisites check ────────────────────────────────────────────────────────

python_version=$(python3 -c 'import sys; print(sys.version_info[:2])' 2>/dev/null || echo "(0, 0)")
if python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
    echo "Python $(python3 --version) — OK"
else
    echo "ERROR: Python 3.10 or later is required (found: $(python3 --version 2>/dev/null || echo 'none'))" >&2
    exit 1
fi

if ! command -v mqlib &>/dev/null; then
    echo ""
    echo "WARNING: 'mqlib' not found on PATH."
    echo "  Install MQLib from https://github.com/MQLib/MQLib and add the binary to your PATH"
    echo "  before running benchmarks. Continuing without it."
    echo ""
fi

# ── Git submodules ─────────────────────────────────────────────────────────────

echo "Initialising git submodules..."
# Note: pangene_fork and openqaoa require Sanger internal GitLab SSH access.
# If you are outside Sanger, those submodules will fail — this is expected.
# All other submodules are publicly accessible.
git submodule update --init --recursive 2>&1 | grep -v '^$' || true

# ── Python packages ────────────────────────────────────────────────────────────

echo ""
echo "Installing Python packages (editable)..."
pip install -e qubo_solvers/
pip install -e new_qubo_formulation/
pip install -e new_hubo_formulation/
pip install -e qiskit_simulation/

# ── Post-install reminders ─────────────────────────────────────────────────────

echo ""
echo "========================================================================"
echo "Installation complete. Two manual steps remain:"
echo ""
echo "1. Output paths"
echo "   Edit qubo_solvers/qubo_solvers/definitions.py and set DATA_DIR and"
echo "   OUT_DIR to directories that exist on your machine. The defaults point"
echo "   to Sanger Lustre scratch and will not work elsewhere."
echo ""
echo "2. Solver credentials"
echo "   Gurobi:  obtain a licence at https://support.gurobi.com/"
echo "   D-Wave:  create an account at https://cloud.dwavesys.com/leap/"
echo "            then run:  dwave config create"
echo "========================================================================"
