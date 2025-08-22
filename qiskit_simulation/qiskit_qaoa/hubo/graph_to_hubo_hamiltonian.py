import numpy as np
import networkx as nx
from functools import reduce
from qiskit.quantum_info import SparsePauliOp
from qiskit_qaoa.utils.hamiltonian_utils import indices_to_pauli
from qiskit_qaoa.utils.string_utils import bin_rep

rng = np.random.default_rng(seed=1)

# TODO: could map all binary values corresponding to >= V to the "end" node, eliminating the need for Grover mixers
# at the cost of increasing the number of interactions

def graph_to_hubo_hamiltonian(
        graph: nx.Graph, n: int, T: int, lamda: float, fraction_terms: float=1.0
) -> SparsePauliOp:
    nodes = list(graph.nodes)
    V = len(nodes)
    terms_to_keep = rng.choice(T-1, int(np.ceil((T-1) * fraction_terms)), replace=False)
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
                                for j in [nodes.index(nbr) for nbr in graph.neighbors(nodes[i])] + [V]
                            ],
                            SparsePauliOp('I'*n*T, np.array([0]))
                        )
                    )
                    for i in range(V)
                ] + [
                    SparsePauliOp.compose(
                        reduce(
                            SparsePauliOp.compose,
                            [0.5 * (SparsePauliOp('I' * n * T, np.array([1])) + (1 - 2 * bin_rep(V, n)[k]) * indices_to_pauli(t, k, n, T)) for k in range(n)],
                            SparsePauliOp('I'*n*T, np.array([1]))
                        ),
                        reduce(
                            SparsePauliOp.compose,
                            [0.5 * (SparsePauliOp('I' * n * T, np.array([1])) + (1 - 2 * bin_rep(V, n)[k]) * indices_to_pauli(t+1, k, n, T)) for k in range(n)],
                            SparsePauliOp('I'*n*T, np.array([1]))
                        )
                    )
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
    return hamiltonian