"""Construct the HUBO QAOA cost Hamiltonian as a Qiskit ``SparsePauliOp``.

The Hamiltonian contains two types of terms:

* **Objective** – for each pair of orientation nodes ``(i, i+1)`` (segment ``i`` in
  both orientations), penalises configurations in which the sum of indicators
  ``∑_t [node at timestep t ∈ {i, i+1}]`` deviates from the copy number of segment
  ``i``.  This is a squared-penalty term that expands to higher-order Pauli products.

* **Constraint** – for each selected pair of consecutive timesteps ``(t, t+1)``,
  penalises configurations in which the binary-encoded node index at ``t+1`` is *not*
  a graph neighbour of the binary-encoded node index at ``t``.  The projection onto
  node ``i`` at timestep ``t`` is the product
  ``∏_{k=0}^{n-1} ½(I + (1 − 2·b_k(i)) Z_{t,k})``
  where ``b_k(i)`` is the ``k``-th bit of the binary representation of ``i``.

Because these projectors are products of single-qubit operators the resulting Pauli
strings can be 2-, 3-, 4-, … body depending on ``n``, making this a genuine HUBO
(Higher-Order Unconstrained Binary Optimisation) problem.
"""

import numpy as np
import networkx as nx
from functools import reduce
from qiskit.quantum_info import SparsePauliOp
from qiskit_qaoa.utils.hamiltonian_utils import indices_to_pauli
from qiskit_qaoa.utils.string_utils import bin_rep

rng = np.random.default_rng(seed=1)

def graph_to_hubo_hamiltonian(
        graph: nx.Graph, n: int, T: int, lamda: float, constraint_terms: float | tuple[int,...]=1.0
) -> tuple[SparsePauliOp, float]:
    """Build the HUBO cost Hamiltonian for the pangenome tangle resolution problem.

    Constructs a ``SparsePauliOp`` over ``n * T`` qubits.  Qubit ``t * n + k`` is the
    ``k``-th encoding bit for the node index at timestep ``t``.

    The Hamiltonian is:

    .. code-block:: none

        H = λ · H_constraint + H_objective

    **Constraint term** (``H_constraint``): for each selected consecutive timestep
    pair ``(t, t+1)``, the penalty equals the number of timestep-pairs at which the
    binary-encoded node at ``t+1`` is *not* a graph neighbour of the binary-encoded
    node at ``t``.  Formally, for each such pair the contribution is:

    .. code-block:: none

        I − ∑_i Π_{t}(i) · ∑_{j ∈ N(i)} Π_{t+1}(j)

    where ``Π_{t}(i) = ∏_k ½(I + (1−2·b_k(i)) Z_{t,k})`` projects onto node ``i``
    at timestep ``t``.

    **Objective term** (``H_objective``): for each segment ``s`` (node pair ``2s``,
    ``2s+1``), the penalty is:

    .. code-block:: none

        (∑_t [Π_t(2s) + Π_t(2s+1)] − copy_number(s))²

    **Normalisation**: the Hamiltonian is divided by the largest non-identity
    coefficient magnitude so that all coefficients lie in ``[−1, 1]``.

    Args:
        graph: Orientation-aware pangenome DiGraph as returned by
            ``gfa_file_to_graph``.  Nodes must carry a ``weight`` attribute giving
            the copy number.
        n: Number of binary-encoding qubits per timestep, ``⌈log₂(V)⌉``.
        T: Number of timesteps (typically ``total_weight`` from ``gfa_file_to_graph``).
        lamda: Penalty weight ``λ`` scaling the constraint term relative to the
            objective term.
        constraint_terms: Controls which consecutive timestep pairs ``(t, t+1)``
            contribute to the constraint term.

            * ``float`` – fraction of the ``T−1`` available pairs to keep, chosen
              uniformly at random without replacement (e.g. ``1.0`` keeps all pairs,
              ``0.5`` keeps half).
            * ``tuple[int, ...]`` – explicit indices of the timestep pairs to keep
              (each element ``t`` corresponds to the constraint between timesteps
              ``t`` and ``t+1``).

    Returns:
        A two-tuple ``(hamiltonian, norm)`` where:

        * ``hamiltonian`` (``SparsePauliOp``) – the normalised cost Hamiltonian,
          simplified and sorted, with coefficients in ``[−1, 1]``.
        * ``norm`` (``float``) – the largest non-identity coefficient magnitude
          before normalisation, used to rescale energies back to physical units.

    Raises:
        Exception: If ``constraint_terms`` is neither a ``float`` nor a
            ``tuple[int, ...]``.
    """
    nodes = list(graph.nodes)
    V = len(nodes)
    if isinstance(constraint_terms, tuple):
        terms_to_keep = constraint_terms
    elif isinstance(constraint_terms, float):
        terms_to_keep = rng.choice(T-1, int(np.ceil((T-1) * constraint_terms)), replace=False)
    else:
        raise Exception(f'Expected float or tuple of ints for constraint_terms, got {constraint_terms}')
    print(f'Keeping constraints at times: {terms_to_keep}')   
    cons_spo = reduce(
        SparsePauliOp._add,
        [
            SparsePauliOp('I'*n*T, np.array([1])) - reduce(
                SparsePauliOp._add,
                [
                    SparsePauliOp.compose(
                        reduce(
                            SparsePauliOp.compose,
                            [0.5 * (SparsePauliOp('I' * n * T, np.array([1])) + (1 - 2 * bin_rep(i, n)[k]) * indices_to_pauli(t, k, n, T)) for k in range(n)],
                            SparsePauliOp('I'*n*T, np.array([1]))
                        ),
                        reduce(
                            SparsePauliOp._add,
                            [
                                reduce(
                                    SparsePauliOp.compose,
                                    [0.5 * (SparsePauliOp('I' * n * T, np.array([1])) + (1 - 2 * bin_rep(j, n)[k]) * indices_to_pauli(t+1, k, n, T)) for k in range(n)],
                                    SparsePauliOp('I'*n*T, np.array([1]))
                                )
                                for j in [nodes.index(nbr) for nbr in graph.neighbors(nodes[i])]
                            ],
                            SparsePauliOp('I'*n*T, np.array([0]))
                        )
                    )
                    for i in range(V)
                ],
                SparsePauliOp('I'*n*T, np.array([0]))
            ) for t in terms_to_keep
        ],
        SparsePauliOp('I'*n*T, np.array([0]))
    )


    obj_spo = reduce(
        SparsePauliOp._add,
        [
            (
                reduce(
                    SparsePauliOp._add,
                    [
                        reduce(
                            SparsePauliOp.compose,
                            [0.5 * (SparsePauliOp('I' * n * T, np.array([1])) + (1 - 2 * bin_rep(i, n)[k]) * indices_to_pauli(t, k, n, T)) for k in range(n)]
                        ) + reduce(
                            SparsePauliOp.compose,
                            [0.5 * (SparsePauliOp('I' * n * T, np.array([1])) + (1 - 2 * bin_rep(i+1, n)[k]) * indices_to_pauli(t, k, n, T)) for k in range(n)]
                        )
                        for t in range(T)
                    ],
                    SparsePauliOp('I'  * n * T, np.array([0]))
                ) 
                - SparsePauliOp('I' * n * T, graph.nodes[nodes[i]]["weight"]) 
            ) ** 2
            for i in range(0, V, 2)
        ],
        SparsePauliOp('I'  * n * T, np.array([0]))
    )

    hamiltonian = lamda * cons_spo + obj_spo
    hamiltonian = hamiltonian.simplify()
    hamiltonian = hamiltonian.sort()
    norm = np.abs(max(hamiltonian.coeffs[1:]))
    hamiltonian /= norm
    return hamiltonian, norm