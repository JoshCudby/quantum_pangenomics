"""Convert a compiled HUBO circuit with fixed evolution times to a parameterised form.

During compilation (``compilation.py``) each Pauli-evolution gate is given a fixed
evolution time that encodes the corresponding Hamiltonian coefficient.  This produces
a circuit that is optimal for hardware routing but whose cost angle is baked in.

This module re-introduces a single free scalar parameter ``γ`` by replacing every
fixed time ``t_i`` with ``t_i * γ``.  The resulting circuit can then be swept over
``γ`` during the QAOA simulation without recompiling, and is consumed by
``get_LR_qaoa_circuit`` as the ``cost_circuit`` argument.
"""

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


def parameterise_circuit(qc: QuantumCircuit, parameter: Parameter) -> QuantumCircuit:
    """Re-parameterise a compiled circuit by introducing a global scalar QAOA angle.

    The compiled cost circuit produced by ``compilation.py`` contains gates whose
    rotation angles are fixed real numbers encoding the Hamiltonian coefficients.
    This function clones that circuit, replacing each gate parameter ``t_i`` with
    the expression ``t_i * parameter``.  The result has exactly one free
    ``Parameter`` (the supplied ``parameter`` object), which is typically bound to
    the QAOA cost angle ``γ`` at runtime.

    Only circuits in which every gate has at most one parameter are supported; gates
    with no parameters (e.g. SWAP, CX) are copied verbatim.

    Args:
        qc: A compiled ``QuantumCircuit`` whose gate parameters are fixed real
            values (i.e. the circuit has no free ``Parameter`` objects).
        parameter: The Qiskit ``Parameter`` to introduce as the global scalar
            multiplier, typically ``Parameter('γ')``.

    Returns:
        A new ``QuantumCircuit`` on the same number of qubits in which every
        gate parameter ``t_i`` has been replaced by ``t_i * parameter``.

    Raises:
        Exception: If any gate in ``qc`` has more than one parameter.
    """
    clone = QuantumCircuit(qc.num_qubits)

    for g in qc:
        if len(g.operation.params) == 0:
            clone.append(g)
        elif len(g.operation.params) > 1:
            raise Exception(f'Only single parameter circuits allowed, received: {g}')
        else:
            op = g.operation.copy()
            op.params = [g.operation.params[0] * parameter]
            clone.append(op, g.qubits)
    return clone