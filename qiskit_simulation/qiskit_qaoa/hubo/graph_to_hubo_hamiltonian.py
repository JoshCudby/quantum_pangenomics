"""Convert an orientation-aware pangenome graph to a HUBO SparsePauliOp.

Binary encoding: each of the T timesteps uses n = ceil(log2(V)) qubits to
binary-encode the index of the visited node (or orientation-strand).  The
total register therefore contains n * T qubits.  This encoding is more
qubit-efficient than a one-hot scheme but produces higher-order (3-body,
4-body, ...) Pauli-Z interaction terms in the Hamiltonian.

The Hamiltonian has the form:

    H = lambda * H_constraint + H_objective

where:

    H_constraint  penalises transitions that are not valid graph edges
                  (or transitions to/from "phantom" indices in [V, 2^n)).

    H_objective   is a squared-coverage term: it penalises paths whose
                  visit-count to each pair of adjacent orientation-strands
                  deviates from the target copy-number weight stored on the
                  corresponding graph node.

The returned SparsePauliOp is simplified and lexicographically sorted.
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
) -> SparsePauliOp:
    """Build the HUBO Hamiltonian for pangenome tangle resolution via QAOA.

    The binary encoding maps each timestep t in {0, ..., T-1} to n qubits
    (qubit indices t*n to t*n + n - 1) that store the binary representation of
    the visited node index.  Because n = ceil(log2(2*V+1)) in practice, indices
    in [V, 2^n) are "phantom" states that must also be treated as valid
    absorbers in the constraint term to avoid spurious penalties.

    Constraint term H_constraint:
        For each selected pair of adjacent timesteps (t, t+1) the term
        penalises bitstrings in which the node at time t is NOT a valid
        in-graph neighbour of the node at time t+1.  The penalty is built by
        projecting each qubit register onto a specific node index via the
        product-of-projectors construction::

            P(t, i) = prod_{k=0}^{n-1}  (I + (1 - 2*b_k(i)) * Z_{t,k}) / 2

        where b_k(i) is the k-th bit of the binary representation of index i.
        Each timestep-pair contributes one term of the form::

            I - sum_{i in [0,V)} P(t,i) * sum_{j in neighbours(i) | [V,2^n)} P(t+1,j)
              - sum_{i in [V,2^n)} P(t,i) * sum_{j in [V,2^n)} P(t+1,j)

        This evaluates to zero if and only if the transition (i -> j) is a
        valid graph edge or the path is in a phantom state at both ends.

    Objective term H_objective:
        For each pair of adjacent orientation-strands (nodes 2i, 2i+1),
        the total number of times the path visits either strand across all
        T timesteps is computed and penalised for deviating from the
        target copy-number weight w_i::

            H_obj += (sum_{t} [P(t,2i) + P(t,2i+1)] - w_i)^2

    Args:
        graph: Orientation-aware pangenome graph.  Nodes must carry a
            ``"weight"`` attribute (the target copy number).  Edges encode
            valid strand-to-strand transitions.
        n: Number of encoding qubits per timestep; satisfies n = ceil(log2(V))
            where V = len(graph.nodes).
        T: Number of timesteps (path length).  The total qubit count is n * T.
        lamda: Penalty coefficient for the constraint term.  Typical value: 10.
        constraint_terms: Specifies which timestep-transition pairs to include
            in H_constraint.

            - ``float`` in [0, 1]: fraction of the T-1 transitions to keep,
              chosen uniformly at random (seeded via the module-level ``rng``).
            - ``tuple[int, ...]``: explicit list of timestep indices t for
              which the (t, t+1) constraint is enforced.

    Returns:
        A simplified, sorted SparsePauliOp over n*T qubits representing
        ``lamda * H_constraint + H_objective``.

    Raises:
        Exception: If ``constraint_terms`` is neither a float nor a tuple of
            ints.
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
                                for j in [nodes.index(nbr) for nbr in graph.neighbors(nodes[i])] + list(range(V, 2**n))
                            ],
                            SparsePauliOp('I'*n*T, np.array([0]))
                        )
                    )
                    for i in range(V)
                ] + [
                    SparsePauliOp.compose(
                        reduce(
                            SparsePauliOp.compose,
                            [0.5 * (SparsePauliOp('I' * n * T, np.array([1])) + (1 - 2 * bin_rep(ii, n)[k]) * indices_to_pauli(t, k, n, T)) for k in range(n)],
                            SparsePauliOp('I'*n*T, np.array([1]))
                        ),
                        reduce(
                            SparsePauliOp._add,
                            [
                                reduce(
                                    SparsePauliOp.compose,
                                    [0.5 * (SparsePauliOp('I' * n * T, np.array([1])) + (1 - 2 * bin_rep(jj, n)[k]) * indices_to_pauli(t+1, k, n, T)) for k in range(n)],
                                    SparsePauliOp('I'*n*T, np.array([1]))
                                )
                                for jj in range(V, 2**n)
                            ],
                            SparsePauliOp('I'*n*T, np.array([0]))
                        )
                    )
                    for ii in range(V, 2**n)
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
    return hamiltonian