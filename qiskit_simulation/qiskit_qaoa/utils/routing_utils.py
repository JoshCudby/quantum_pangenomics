"""Low-level gate routing utilities: greedy parity reduction and Gaussian elimination.

Implements two families of algorithms used to decompose a set of multi-qubit
Pauli-Z rotation gates into a sequence of CX and RZ gates on a quantum circuit:

- ``greedy_parity_network`` / ``hardware_greedy_parity_network``: reduce the
  parity vectors associated with the Pauli terms one at a time using CX gates
  chosen greedily by Hamming weight, producing an upper-triangular parity
  matrix.
- ``greedy_gaussian_elimination``: perform additional row operations over GF(2)
  to reduce the resulting matrix to the identity, generating the CX uncompute
  sequence.
- ``greedy_reduce_parity``: BFS-based helper that finds the shortest CX path
  to gather all parity information at a single qubit on a restricted coupling
  map.
"""

import numpy as np
from collections import deque, defaultdict
from typing import List, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from itertools import combinations
from qiskit.transpiler.coupling import CouplingMap


def weight(s: tuple[int,...] | list[int] | np.ndarray) -> int:
    """Return the Hamming weight (number of ones) of a binary vector.

    Args:
        s: A tuple, list, or numpy array of integers assumed to be in ``{0, 1}``.

    Returns:
        The sum of the elements (i.e. the number of ones).
    """
    return sum(s)


def greedy_parity_network(
    current_layer: dict[tuple[int,...], Gate],
    quantum_circuit: QuantumCircuit
):
    """Decompose a layer of multi-qubit Z-rotation gates using greedy parity reduction.

    Weight-1 (single-qubit) gates are applied as direct RZ rotations.  The
    remaining higher-weight gates are reduced by repeatedly choosing the lowest-
    weight parity vector and emitting a CX gate between its two lowest-indexed
    support qubits to fold it into a lower-weight vector.  The process
    terminates when all remaining parity vectors have weight 1.

    Args:
        current_layer: A mapping from binary parity tuples (one int per qubit,
            indicating which qubits contribute to the Pauli term) to the
            corresponding ``PauliEvolutionGate`` to apply.
        quantum_circuit: The ``QuantumCircuit`` to which CX and RZ gates are
            appended in-place.

    Returns:
        The parity matrix ``A`` (numpy ndarray, shape ``(n, n)``) representing
        the accumulated row operations over GF(2).  This can be passed to
        ``greedy_gaussian_elimination`` to generate the uncompute sequence.
    """
    S = list(current_layer.keys())
    n = len(S[0]) 
    A = np.eye(n,n)
    for s in S:
        if weight(s) == 1:
                # print(f'Removing: {s}')
                coeff = 2 * np.real_if_close(current_layer[s].params)[0]
                quantum_circuit.rz(coeff, s.index(1))            
    S = [s[:] for s in S if weight(s) > 1]
    S.sort(key=weight)
    while len(S) > 0: 
        y = S[0] 
        # print(f'Interaction: {y}') 
        indices = [i for i in range(n) if y[i] == 1] 
        # print(f'CX: {indices[0], indices[1]}') 
        quantum_circuit.cx(indices[0], indices[1]) 
        A[indices[1]] = (A[indices[1]] + A[indices[0]]) % 2
        new_layer = {}
        for s in S: 
            ss = list(s[:])
            ss[indices[0]] = (s[indices[0]] + s[indices[1]]) % 2 
            ss = tuple(ss)
            if weight(ss) > 1:
                new_layer[ss] = current_layer[s]
            else: 
                # print(f'Removing: {ss}')
                coeff = 2 * np.real_if_close(current_layer[s].params)[0]
                quantum_circuit.rz(coeff, ss.index(1))
        current_layer = new_layer
        S = list(current_layer.keys())
        S = [s[:] for s in S if weight(s) > 1]
        S.sort(key=weight)
    return A


def greedy_gaussian_elimination(A: np.ndarray, quantum_circuit: QuantumCircuit):
    """Reduce a GF(2) parity matrix to the identity using greedy row operations.

    At each step the pair of rows ``(i, j)`` is chosen whose XOR produces the
    greatest reduction in the maximum row weight (i.e. the greedy criterion).
    The corresponding CX gate is appended to ``quantum_circuit``.  This
    generates the uncompute sequence needed to restore the identity parity
    matrix after ``greedy_parity_network``.

    Args:
        A: An ``(n, n)`` numpy array with entries in ``{0, 1}`` representing
            the current GF(2) parity matrix.  Modified in-place.
        quantum_circuit: The ``QuantumCircuit`` to which CX gates are appended
            in-place.

    Raises:
        Exception: If no beneficial row operation can be found (should not
            occur for a full-rank matrix).
    """
    n = A.shape[0]
    while any(np.sum(np.abs(A), axis=0) > 1) :
        C = combinations(range(n), 2)
        best_score = 0
        l, m = None, None
        for (i, j) in C:
            new_score = max(weight(A[i, :]), weight(A[j, :])) - weight((A[i,:]+A[j,:]) % 2)
            if new_score > best_score:
                l, m = i, j
                best_score = new_score
        if l is None or m is None:
            raise Exception('No row op found')
        # print(f'CX qubits: {l, m}')
        if weight(A[l, :]) < weight(A[m, :]):
            quantum_circuit.cx(l, m)
            A[m, :] = (A[l, :] + A[m, :]) % 2
        else:
            quantum_circuit.cx(m, l)
            A[l, :] = (A[l, :] + A[m, :]) % 2            
    return


def greedy_reduce_parity(
    parity: List[int],
    coupling_map: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """
    Greedily reduce a binary parity vector to Hamming weight 1 using CX gates.

    A CX(c, t) updates parity[t] ^= parity[c].

    Returns:
        A list of (control, target) CX gates.
    """
    parity = parity[:]

    graph = defaultdict(set)
    for a, b in coupling_map:
        graph[a].add(b)
        graph[b].add(a)

    def shortest_path(src: int, dst: int) -> List[int]:
        """BFS shortest path in the coupling graph."""
        q = deque([src])
        prev = {src: None}

        while q:
            u = q.popleft()
            if u == dst:
                break
            for v in graph[u]:
                if v not in prev:
                    prev[v] = u
                    q.append(v)

        if dst not in prev:
            raise ValueError(f"No path between {src} and {dst} in the coupling graph.")

        path = []
        cur = dst
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        return path[::-1]

    active = [i for i, p in enumerate(parity) if p]
    if len(active) <= 1:
        return []

    collector = active[0]
    gates: List[Tuple[int, int]] = []

    while sum(parity) > 1:
        candidates = [i for i, p in enumerate(parity) if p and i != collector]
        if not candidates:
            break

        path = min((shortest_path(collector, t) for t in candidates), key=len)

        u = path[0]
        for v in path[1:]:
            if parity[v] == 0:
                gates.append((u, v))
                parity[v] ^= parity[u]

                gates.append((v, u))
                parity[u] ^= parity[v]

                u = v
            else:
                gates.append((u, v))
                parity[v] ^= parity[u]

        collector = u

    return gates



def hardware_greedy_parity_network(
    current_layer: dict[tuple[int,...], Gate],
    quantum_circuit: QuantumCircuit,
    coupling_map: CouplingMap
):
    """Decompose a multi-qubit Z-rotation layer respecting a hardware coupling map.

    Analogous to ``greedy_parity_network`` but uses BFS over the coupling
    graph (via ``greedy_reduce_parity``) to route each CX gate through
    intermediate qubits when the two target qubits are not directly connected.
    Tracks the full CX sequence applied and returns it for uncomputing.

    Args:
        current_layer: Mapping from binary parity tuples to the corresponding
            ``PauliEvolutionGate``.
        quantum_circuit: The ``QuantumCircuit`` to which gates are appended
            in-place.
        coupling_map: A Qiskit ``CouplingMap`` defining which qubit pairs may
            receive a direct CX gate.

    Returns:
        A list of ``(control, target)`` tuples representing the CX gates
        that were applied, in order.  This is needed to uncompute the parity
        transformation.
    """
    S = list(current_layer.keys())
    n = len(S[0]) 
    cx_gates = []
    for s in S:
        if weight(s) == 1:
                # print(f'Removing: {s}')
                coeff = 2 * np.real_if_close(current_layer[s].params)[0]
                quantum_circuit.rz(coeff, s.index(1))            
    S = [s[:] for s in S if weight(s) > 1]
    S.sort(key=weight)
    """
    Instead of applying cx(I[0],I[1]), compute the full CX network needed to implement y.
    Should always be possible without SWAPs.
    Similar logic as the compute site in original implementations.
    After each CX, update A and check impossible_gates to see if any are weight 1.
    After all CX, remove y.
    Track and reverse CXs (greedy elim will fail as not UT and can't apply long range)
    """
    while len(S) > 0: 
        y = S[0] 
        cx_gates_to_apply = greedy_reduce_parity(y, list(coupling_map))
        # print(f'Interaction: {y}') 
        indices = [i for i in range(n) if y[i] == 1] 
        # print(f'CX: {indices[0], indices[1]}') 
        quantum_circuit.cx(indices[0], indices[1]) 
        cx_gates.append((indices[0], indices[1]))
        new_layer = {}
        for s in S: 
            ss = list(s[:])
            ss[indices[0]] = (s[indices[0]] + s[indices[1]]) % 2 
            ss = tuple(ss)
            if weight(ss) > 1:
                new_layer[ss] = current_layer[s]
            else: 
                # print(f'Removing: {ss}')
                coeff = 2 * np.real_if_close(current_layer[s].params)[0]
                quantum_circuit.rz(coeff, ss.index(1))
        current_layer = new_layer
        S = list(current_layer.keys())
        S = [s[:] for s in S if weight(s) > 1]
        S.sort(key=weight)
    return cx_gates