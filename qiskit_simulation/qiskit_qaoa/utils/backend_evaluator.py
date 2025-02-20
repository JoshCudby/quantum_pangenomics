from __future__ import annotations
from collections.abc import Callable
import numpy as np
import rustworkx as rx

from qiskit.transpiler import CouplingMap
from qiskit.providers import Backend


TWO_Q_GATES = ["cx", "ecr", "cz"]


def find_lines(length: int, backend: Backend) -> list[int]:
    """Finds all possible lines of length `length` for a specific backend topology.

    This method can take quite some time to run on large devices since there
    are many paths.

    Returns:
        The found paths.
    """

    if backend.version == 2:
        coupling_map = CouplingMap(backend.coupling_map)
    else:
        coupling_map = CouplingMap(backend.configuration().coupling_map)

    if not coupling_map.is_symmetric:
        coupling_map.make_symmetric()

    all_paths = rx.all_pairs_all_simple_paths(
        coupling_map.graph,
        min_depth=length,
        cutoff=length,
    ).values()

    paths = np.asarray(
        [
            (list(c), list(sorted(list(c))))
            for a in iter(all_paths)
            for b in iter(a)
            for c in iter(a[b])
        ]
    )

    # filter out duplicated paths
    _, unique_indices = np.unique(paths[:, 1], return_index=True, axis=0)
    paths = paths[:, 0][unique_indices].tolist()

    return paths


def evaluate_fidelity(path: list[int], backend: Backend, edges: rx.EdgeList) -> float:
    """Evaluates fidelity on a given list of qubits based on the two-qubit gate error
    for a specific backend.

    Returns:
       Path fidelity.
    """

    two_qubit_fidelity = {}

    if backend.version == 2:
        target = backend.target
        try:
            gate_name = list(set(TWO_Q_GATES).intersection(backend.operation_names))[0]
        except IndexError as exc:
            raise ValueError("Could not identify two-qubit gate") from exc

        for edge in edges:
            try:
                cx_error = target[gate_name][edge].error
            except: # noqa: E722
                cx_error = target[gate_name][edge[::-1]].error

            two_qubit_fidelity[tuple(edge)] = 1 - cx_error
    else:
        props = backend.properties()
        try:
            gate_name = list(set(TWO_Q_GATES).intersection(backend.configuration().basis_gates))[0]
        except IndexError as exc:
            raise ValueError("Could not identify two-qubit gate") from exc

        for edge in edges:
            try:
                cx_error = props.gate_error(gate_name, edge)
            except: # noqa: E722
                cx_error = props.gate_error(gate_name, edge[::-1])

            two_qubit_fidelity[tuple(edge)] = 1 - cx_error

    if not path or len(path) == 1:
        return 0.0

    fidelity = 1.0
    for idx in range(len(path) - 1):
        fidelity *= two_qubit_fidelity[(path[idx], path[idx + 1])]
    return fidelity


class BackendEvaluator:
    """
    Finds best subset of qubits for a given device that maximizes a given
    metric for a given geometry.
    This subset can be provided as an initial_layout for the SwapStrategy
    transpiler pass.
    """

    def __init__(self, backend: Backend):
        self.backend = backend
        if backend.version == 2:
            coupling_map = CouplingMap(backend.coupling_map)
        else:
            coupling_map = CouplingMap(backend.configuration().coupling_map)
        self.coupling_map = coupling_map
        if not self.coupling_map.is_symmetric:
            self.coupling_map.make_symmetric()

    def evaluate(
        self,
        num_qubits: int,
        subset_finder: Callable | None = None,
        metric_eval: Callable | None = None,
    ):
        """
        Args:
            num_qubits: the number of qubits
            subset_finder: callable, will default to "find_line"
            metric_eval: callable, will default to "evaluate_fidelity"
        """

        if metric_eval is None:
            metric_eval = evaluate_fidelity

        if subset_finder is None:
            subset_finder = find_lines

        # TODO: add callbacks
        qubit_subsets = subset_finder(num_qubits, self.backend)

        # evaluating the subsets
        scores = [
            metric_eval(subset, self.backend, self.coupling_map.get_edges())
            for subset in qubit_subsets
        ]

        # Return the best subset sorted by score
        best_subset, best_score = min(zip(qubit_subsets, scores), key=lambda x: -x[1])
        num_subsets = len(qubit_subsets)

        return best_subset, best_score, num_subsets