"""Q matrix to Ising Hamiltonian conversion and interaction graph utilities.

Provides helpers for loading QUBO problem data, constructing the equivalent
Ising Hamiltonian as a ``SparsePauliOp``, and extracting the interaction
structure needed by the QAOA routing infrastructure.
"""

import numpy as np
import networkx as nx
import re
import pickle
from qiskit.quantum_info import SparsePauliOp
from qiskit_optimization import QuadraticProgram


rng = np.random.default_rng(seed=1)


def get_qp(data_file) -> QuadraticProgram:
    """Load a QUBO problem from a numpy file and return it as a QuadraticProgram.

    Reads Q and offset from a ``.npy`` file, normalises Q by its maximum
    absolute entry, and constructs a ``QuadraticProgram`` minimisation problem.

    Args:
        data_file: Path to a numpy ``.npy`` file whose first element is the
            Q matrix and second is the constant offset.

    Returns:
        A ``QuadraticProgram`` with binary variables and the normalised QUBO
        objective.
    """
    data = np.load(data_file, allow_pickle=True)
    Q, offset, _, _  = data
    Q = np.triu(Q) * 2
    Q -= np.triu(np.triu(Q).T) / 2

    normalisation = np.max(np.abs(Q))
    Q = Q / normalisation
    offset = offset / normalisation

    mod = QuadraticProgram("QUBO test")
    mod.binary_var_list(Q.shape[0])
    mod.minimize(constant=offset, linear=None, quadratic=Q)
    return mod


def get_Q_and_hamiltonian(data_file):
    """Load a QUBO problem from a pickle file and convert to an Ising Hamiltonian.

    Does not normalise Q or the Hamiltonian.

    Args:
        data_file: Path to a pickle file with keys ``'Q'`` and ``'offset'``.

    Returns:
        A tuple ``(Q, hamiltonian, offset, ising_offset)`` where ``Q`` is the
        upper-triangular QUBO matrix, ``hamiltonian`` is a ``SparsePauliOp``
        sorted by weight, and ``ising_offset`` is the constant from the
        QUBO-to-Ising mapping.
    """
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    Q = data['Q']
    offset = data['offset']
    
    Q = np.triu(Q) * 2
    Q -= np.triu(np.triu(Q).T) / 2


    mod = QuadraticProgram("QUBO test")
    mod.binary_var_list(Q.shape[0])
    mod.minimize(constant=offset, linear=None, quadratic=Q)
    hamiltonian, ising_offset = mod.to_ising()
    hamiltonian = hamiltonian.sort(weight=True)
    return Q, hamiltonian, offset, ising_offset


def get_normalised_Q_and_hamiltonian(data_file):
    """Load a QUBO problem, build its Ising Hamiltonian, and normalise both.

    The Q matrix is symmetrised to upper-triangular form before conversion.
    The Hamiltonian coefficients are normalised by the largest absolute
    coefficient so that all weights lie in [-1, 1].

    Args:
        data_file: Path to a pickle file containing a dict with keys ``'Q'``
            (numpy ndarray, QUBO matrix) and ``'offset'`` (float, constant
            energy offset).

    Returns:
        A tuple ``(Q, hamiltonian, offset, ising_offset, normalisation,
        ham_norm)`` where:

        - ``Q``: Normalised upper-triangular QUBO matrix.
        - ``hamiltonian``: Normalised ``SparsePauliOp`` Ising Hamiltonian,
          sorted by weight.
        - ``offset``: Normalised constant term of the QUBO objective.
        - ``ising_offset``: Constant shift introduced by the QUBO-to-Ising
          transformation.
        - ``normalisation``: Maximum absolute entry of the raw Q matrix used
          to normalise Q.
        - ``ham_norm``: Maximum absolute coefficient of the pre-normalised
          Hamiltonian used to normalise it.
    """
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    Q = data['Q']
    offset = data['offset']
    
    Q = np.triu(Q) * 2
    Q -= np.triu(np.triu(Q).T) / 2

    mod = QuadraticProgram("QUBO test")
    mod.binary_var_list(Q.shape[0])
    mod.minimize(constant=offset, linear=None, quadratic=Q)
    hamiltonian, ising_offset = mod.to_ising()
    hamiltonian = hamiltonian.sort(weight=True)

    normalisation = np.max(np.abs(Q))
    Q = Q / normalisation
    offset = offset / normalisation

    ham_norm = np.abs(max(hamiltonian.coeffs))
    hamiltonian /= ham_norm
    ising_offset /= ham_norm
    return Q, hamiltonian, offset, ising_offset, normalisation, ham_norm


def get_objective_and_hamiltonian(data_file):
    """Load and normalise a QUBO problem, returning the objective and Ising Hamiltonian.

    Args:
        data_file: Path to a pickle file with keys ``'Q'`` and ``'offset'``.

    Returns:
        A tuple ``(objective, hamiltonian, ising_offset)`` where ``objective``
        is the ``QuadraticObjective`` of the normalised problem,
        ``hamiltonian`` is the ``SparsePauliOp``, and ``ising_offset`` is the
        constant energy shift.
    """
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    Q = data['Q']
    offset = data['offset']
    
    Q = np.triu(Q) * 2
    Q -= np.triu(np.triu(Q).T) / 2

    normalisation = np.max(np.abs(Q))
    Q = Q / normalisation
    offset = offset / normalisation

    mod = QuadraticProgram("QUBO test")
    mod.binary_var_list(Q.shape[0])
    mod.minimize(constant=offset, linear=None, quadratic=Q)
    hamiltonian, ising_offset = mod.to_ising()
    hamiltonian = hamiltonian.sort(weight=True)
    return mod.objective, hamiltonian, ising_offset


def get_offset(data_file):
    """Return the normalised QUBO constant offset from a numpy data file.

    Args:
        data_file: Path to a ``.npy`` file (same format as ``get_qp``).

    Returns:
        The normalised constant energy offset (float).
    """
    data = np.load(data_file, allow_pickle=True)
    Q, offset, _, _  = data
    Q = np.triu(Q) * 2
    Q -= np.triu(np.triu(Q).T) / 2

    normalisation = np.max(np.abs(Q))
    Q = Q / normalisation
    offset = offset / normalisation
    return offset


def get_ising_offset(data_file):
    """Return the Ising constant offset (after QUBO-to-Ising conversion) from a pickle file.

    Args:
        data_file: Path to a pickle file with keys ``'Q'`` and ``'offset'``.

    Returns:
        The constant energy shift introduced by the QUBO-to-Ising
        transformation (float).
    """
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    Q = data['Q']
    offset = data['offset']
    Q = np.triu(Q) * 2
    Q -= np.triu(np.triu(Q).T) / 2

    normalisation = np.max(np.abs(Q))
    Q = Q / normalisation
    offset = offset / normalisation

    mod = QuadraticProgram("QUBO test")
    mod.binary_var_list(Q.shape[0])
    mod.minimize(constant=offset, linear=None, quadratic=Q)
    _, ising_offset = mod.to_ising()
    return ising_offset


def monomial_to_pauli(monomial, size):
    """Convert a QUBO monomial to a Pauli Z string of length ``size``.

    Args:
        monomial: A monomial object from ``qiskit_optimization`` whose
            ``atoms()`` carry a ``name`` attribute containing the variable
            index as an integer suffix.
        size: Total number of qubits (length of the Pauli string).

    Returns:
        A Pauli string (str) of length ``size`` with ``'Z'`` at the positions
        corresponding to the monomial's variables and ``'I'`` elsewhere.
    """
    indices = [int(re.search(r'[0-9]+', atom.name).group(0)) for atom in monomial.atoms()]
    pauli_str = ['I'] * size
    for i in indices:
        pauli_str[i] = 'Z'
    return ''.join(pauli_str)


def indices_to_pauli(t: int, k: int, n: int, T: int):
    """Build a single-qubit Z SparsePauliOp for variable (t, k) in an n×T register.

    Args:
        t: Time step index (0-indexed).
        k: Node index within time step (0-indexed).
        n: Number of nodes per time step.
        T: Number of time steps (total circuit depth).

    Returns:
        A ``SparsePauliOp`` with a single ``Z`` operator on qubit ``t*n + k``
        and identity everywhere else, with coefficient 1.
    """
    p = ['I'] * n * T
    p[t*n + k] = 'Z'
    return SparsePauliOp(''.join(p), np.array([1]))


def hamiltonian_to_doubles_graph(hamiltonian: SparsePauliOp) -> nx.Graph:
    """Build a weighted interaction graph from the two-qubit ZZ terms of a Hamiltonian.

    Each ZZ term ``Z_i Z_j`` in the Hamiltonian becomes a weighted edge
    ``(i, j)`` in the returned graph.  Single-qubit (Z) and higher-order terms
    are ignored.  All qubits are included as nodes regardless of whether they
    participate in any interaction.

    Args:
        hamiltonian: A ``SparsePauliOp`` Ising Hamiltonian, typically produced
            by ``get_normalised_Q_and_hamiltonian``.

    Returns:
        A ``networkx.Graph`` whose nodes are qubit indices ``0..n-1`` and
        whose edges carry a ``weight`` attribute equal to the corresponding
        ZZ coefficient.
    """
    edges = []
    weights = []
    for t in hamiltonian:
        if np.sum(t.paulis[0].z) == 2:
            edge = np.nonzero(t.paulis[0].z)[0]
            edges.append(edge)
            weights.append(t.coeffs[0])
            
    program_graph = nx.Graph()
    for i in range(hamiltonian.num_qubits):
        program_graph.add_node(i)
    for idx in range(len(weights)):
        program_graph.add_edge(edges[idx][0],edges[idx][1],weight=weights[idx])
    return program_graph


def hamiltonian_to_interactions(hamiltonian: SparsePauliOp, fraction_4=0.0, fraction_6=0.8) -> list[tuple]:
    """Extract a (possibly sampled) list of multi-qubit interaction tuples from a Hamiltonian.

    Iterates over all Pauli terms and collects those with two to six non-identity
    Z operators, applying stochastic subsampling controlled by ``fraction_4``
    and ``fraction_6``.  Useful for constructing the interaction graph passed to
    the SAT mapper or swap-strategy router.

    Args:
        hamiltonian: A ``SparsePauliOp`` Ising Hamiltonian.
        fraction_4: Probability of *skipping* a term with 2–4 Z operators
            (i.e. ``fraction_4=0.0`` keeps all such terms).
        fraction_6: Probability of *skipping* a term with 5–6 Z operators
            (i.e. ``fraction_6=0.8`` keeps roughly 20 % of such terms).

    Returns:
        A list of tuples, each tuple being the sorted qubit indices of one
        interaction retained after sampling.
    """
    interactions: list[tuple] = []
    for t in hamiltonian:
        if np.sum(t.paulis[0].z) < 2 or np.sum(t.paulis[0].z) > 6:
            pass
        elif np.sum(t.paulis[0].z) <= 4 and rng.random() > fraction_4:
            edge = tuple(np.nonzero(t.paulis[0].z)[0])
            interactions.append(edge)
        elif rng.random() > fraction_6:
            edge = tuple(np.nonzero(t.paulis[0].z)[0])
            interactions.append(edge)
    return interactions