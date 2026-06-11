"""Select a topology-matched swap strategy for routing HUBO Pauli-evolution gates.

Wraps ``ExtendedSwapStrategy`` factory methods to provide a single interface that
selects an appropriate qubit topology for the ``n * T`` virtual qubits used in a
HUBO QAOA circuit.  The resulting ``ExtendedSwapStrategy`` is consumed downstream by
``CommutingGateRouterPrecomputeRzz`` (or ``CommutingGateRouterRzz``) to route
multi-qubit Pauli-evolution gates onto the chosen connectivity graph.
"""

from qiskit_qaoa.utils.transpiler_passes import ExtendedSwapStrategy


def get_swap_strategy(coupling_map: str, n: int, T: int):
    """Return an ``ExtendedSwapStrategy`` suited to the requested topology.

    The strategy governs how SWAP gates are inserted to route multi-qubit Pauli
    interactions onto the physical coupling graph.  The virtual circuit uses
    ``n * T`` qubits, and the physical device must be large enough to host them all.

    Supported topologies:

    * ``'line'`` – linear nearest-neighbour chain of ``n * T`` qubits.
    * ``'grid'`` – 2-D grid of shape ``(n, T)``.
    * ``'all'`` – all-to-all connectivity on ``n * T`` qubits (no SWAPs required).
    * ``'heavy-hex'`` – IBM heavy-hexagon topology.  The smallest heavy-hex lattice
      of shape ``(rows, cols)`` satisfying
      ``4 * (rows + cols + rows * cols) ≥ n * T`` is selected automatically, growing
      rows and columns alternately to keep the aspect ratio balanced.

    Args:
        coupling_map: One of ``'line'``, ``'grid'``, ``'all'``, or ``'heavy-hex'``.
        n: Number of binary-encoding qubits per timestep.
        T: Number of timesteps; the total virtual qubit count is ``n * T``.

    Returns:
        An ``ExtendedSwapStrategy`` instance configured for the requested topology.

    Raises:
        Exception: If ``coupling_map`` is not one of the recognised options.
    """
    match coupling_map:
        case 'line':
            extended_swap_strat = ExtendedSwapStrategy.from_line(list(range(n * T)))
        case 'grid':
            extended_swap_strat = ExtendedSwapStrategy.from_grid(n, T)
        case 'all':
            extended_swap_strat = ExtendedSwapStrategy.from_all_to_all(n * T)
        case 'heavy-hex':
            rows, cols = 1, 1
            while 4 * (rows + cols + rows * cols) < n * T:
                if rows < cols:
                    rows += 1
                else:
                    cols += 1
            print(f'Min size to support virtual qubits: {(rows, cols)}')
            extended_swap_strat = ExtendedSwapStrategy.from_heavy_hex(rows, cols)
        case _:
            raise Exception(f'Could not create coupling map: {coupling_map}')
    return extended_swap_strat