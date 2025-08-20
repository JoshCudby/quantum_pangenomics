import numpy as np
import networkx as nx
from functools import reduce
from qiskit.quantum_info import SparsePauliOp
from qiskit_qaoa.utils.hamiltonian_utils import indices_to_pauli
from qiskit_qaoa.utils.string_utils import bin_rep


def graph_to_hubo_hamiltonian(graph: nx.Graph, n: int, N: int, T: int, lamda: float) -> SparsePauliOp:
    nodes = list(graph.nodes)
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
                                for j in [nodes.index(nbr) for nbr in graph.neighbors(nodes[i])] + [N]
                            ],
                            SparsePauliOp('I'*n*T, np.array([0]))
                        )
                    )
                    for i in range(N)
                ] + [
                    SparsePauliOp.compose(
                        reduce(
                            SparsePauliOp.compose,
                            [0.5 * (SparsePauliOp('I' * n * T, np.array([1])) + (1 - 2 * bin_rep(N, n)[k]) * indices_to_pauli(t, k, n, T)) for k in range(n)],
                            SparsePauliOp('I'*n*T, np.array([1]))
                        ),
                        reduce(
                            SparsePauliOp.compose,
                            [0.5 * (SparsePauliOp('I' * n * T, np.array([1])) + (1 - 2 * bin_rep(N, n)[k]) * indices_to_pauli(t+1, k, n, T)) for k in range(n)],
                            SparsePauliOp('I'*n*T, np.array([1]))
                        )
                    )
                ],
                SparsePauliOp('I'*n*T, np.array([0]))
            ) for t in range(T-1) 
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
            for i in range(0, N, 2)
        ],
        SparsePauliOp('I'  * n * T, np.array([0]))
    )

    hamiltonian = lamda * cons_spo + obj_spo
    hamiltonian = hamiltonian.simplify()
    return hamiltonian