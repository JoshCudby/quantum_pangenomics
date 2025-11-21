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
    return hamiltonian