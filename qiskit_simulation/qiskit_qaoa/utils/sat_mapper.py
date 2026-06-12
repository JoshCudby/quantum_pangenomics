"""SAT-based qubit placement for higher-order (HUBO) interactions.

Provides ``HigherOrderSatMapper``, which uses a MaxSAT or SAT solver to find
an optimal qubit layout for a set of multi-qubit Pauli-Z interactions on a
given swap strategy.

The layout problem is formulated as a subgraph isomorphism: each interaction
tuple must map to a set of qubits that form a connected subgraph of the
coupling map after at most ``num_layers`` swap steps (i.e. are within distance
``num_layers`` in the swap strategy distance tensor).  SAT is used rather than
a heuristic mapper because it guarantees optimality — it finds the minimum
number of swap layers such that all required interactions are routable.

Two methods are provided:

- ``hubo_max_sat``: encodes the problem as a Weighted Partial MaxSAT instance
  and invokes the external NuWLS solver; useful when the hard constraints
  (one-to-one mapping) must always be satisfied but some soft adjacency
  constraints can be violated.
- ``find_hubo_mappings``: uses a standard SAT solver (PySAT) with binary
  search over the number of swap layers; all constraints are hard.
"""

from __future__ import annotations

from itertools import combinations, product
from threading import Timer
import numpy as np
import subprocess
import uuid
import os

from pysat.formula import CNF, IDPool, WCNF
from pysat.solvers import Solver

from qopt_best_practices.sat_mapping import SATMapper
from qopt_best_practices.sat_mapping.sat_mapper import SATResult

from qiskit_qaoa.utils.transpiler_passes import ExtendedSwapStrategy
from qiskit_qaoa.utils.logging import get_logger

logger = get_logger(__name__)


def get_cnfs(
    num_nodes_g1: int,
    num_nodes_g2: int,
    program_interactions: list[tuple[int,...]],
    swap_strategy: ExtendedSwapStrategy,
    num_layers: int,
    variables: np.ndarray
):
    """Build one-to-one mapping and adjacency CNF clause lists for SAT placement.

    Constructs two groups of CNF clauses encoding the subgraph-isomorphism
    problem for qubit placement:

    - ``cnf1``: At-least-one and at-most-one clauses ensuring that every
      program node maps to exactly one hardware qubit and vice-versa.
    - ``cnf2``: Adjacency clauses ensuring that each interaction tuple maps
      to a set of hardware qubits that are mutually reachable within
      ``num_layers`` swap steps (using the distance tensor of
      ``swap_strategy``).

    Args:
        num_nodes_g1: Number of program qubits (rows of ``variables``).
        num_nodes_g2: Number of hardware qubits (columns of ``variables``).
        program_interactions: List of interaction tuples to route; sorted
            internally by arity before processing.
        swap_strategy: The ``ExtendedSwapStrategy`` providing pairwise and
            higher-order distance tensors.
        num_layers: Maximum number of swap layers; interactions that require
            more layers are excluded from the adjacency clauses.
        variables: A ``(num_nodes_g1, num_nodes_g2)`` integer array of SAT
            variable IDs.  Entry ``variables[i, j]`` is the variable asserting
            that program qubit ``i`` maps to hardware qubit ``j``.

    Returns:
        A tuple ``(cnf1, cnf2)`` where each element is a list of clauses
        (each clause is a list of signed integer variable IDs, as expected by
        PySAT).
    """
    program_interactions = sorted(program_interactions,key=lambda e: len(e))
        
    # Make a cnf for the one-to-one mapping constraint
    cnf1 = []
    for i in range(num_nodes_g1):
        clause = variables[i, :].tolist()
        cnf1.append(clause)
        for k, m in combinations(clause, 2):
            cnf1.append([-1 * k, -1 * m])
    for j in range(num_nodes_g2):
        clause = variables[:, j].tolist()
        for k, m in combinations(clause, 2):
            cnf1.append([-1 * k, -1 * m])
            
    logger.info(f'Num layers: {num_layers}')

    # Make a cnf for the adjacency constraint
    cnf2 = []
    
    old_num_qubits = 0
    distance_tensor = np.array(0)
    connectivity_tensor = np.array(0)
    for interaction in program_interactions:
        num_qubits = len(interaction)
        if num_qubits != old_num_qubits:
            logger.info('Re-computing distance tensor')
            old_num_qubits = num_qubits
            if num_qubits == 2:
                distance_tensor = swap_strategy.distance_matrix
            else:
                distance_tensor = swap_strategy.distance_tensor(num_qubits)
            connectivity_tensor = ((-1 < distance_tensor) & (distance_tensor <= num_layers)).astype(int)
        
        clause_tensor = np.multiply(connectivity_tensor,  variables[interaction[-1],:])
        fixed_nodes_tensor = np.array([
            [-variables[interaction[i], index_set[i]] for i in range(len(index_set))] for index_set in product(range(num_nodes_g2), repeat=num_qubits-1)
        ]).reshape([num_nodes_g2]*(num_qubits-1)+[num_qubits-1])
        clause = np.concatenate(
            (
                fixed_nodes_tensor,
                clause_tensor,
            ),
            axis=-1,
        ).reshape((num_nodes_g2**(num_qubits-1), num_qubits-1+num_nodes_g2))
        cnf2.extend([c[c != 0].tolist() for c in clause])

    return cnf1, cnf2


class HigherOrderSatMapper(SATMapper):
    """SAT-based qubit placement for HUBO (higher-order) interaction sets.

    Extends ``qopt_best_practices.sat_mapping.SATMapper`` to handle
    interactions involving more than two qubits.  The placement is found by
    solving a (Max)SAT problem where:

    - **Hard constraints** (``cnf1``): the variable assignment must encode a
      one-to-one mapping between program qubits and physical qubits.
    - **Soft constraints** (``cnf2``): for each interaction tuple, at least one
      physical qubit assignment must satisfy the swap-distance constraint.

    Uses the NuWLS-c solver (via subprocess) for MaxSAT and PySAT for standard
    SAT.  The ``timeout`` attribute (inherited from ``SATMapper``) limits solver
    wall time.
    """

    def hubo_max_sat(
        self,
        num_nodes_g1: int,
        program_interactions: list[tuple[int,...]],
        swap_strategy: ExtendedSwapStrategy,
        num_layers: int
    ) -> dict[int, tuple[int, list]] | None:
        """Find a qubit placement using Weighted Partial MaxSAT (NuWLS-c solver).

        Constructs a WCNF formula and calls the external NuWLS-c binary.  Hard
        clauses enforce a one-to-one mapping; soft clauses reward satisfying each
        interaction's adjacency constraint.  The formula, solution, and auxiliary
        files are written to the current working directory and cleaned up after
        the solver finishes.

        Args:
            num_nodes_g1: Number of program qubits (variables) to place.
            program_interactions: List of interaction tuples (sorted by length).
            swap_strategy: The ``ExtendedSwapStrategy`` providing the distance
                tensor.
            num_layers: The maximum number of swap layers to consider.

        Returns:
            A dict ``{num_layers: (cost, mapping)}`` where ``cost`` is the
            MaxSAT objective value and ``mapping`` is a list of
            ``(program_qubit, physical_qubit)`` index pairs; or ``None`` if the
            solver output cannot be parsed.
        """
        num_nodes_g2: int = swap_strategy.distance_matrix.shape[0]
        variable_pool = IDPool(start_from=1)
        variables = np.array(
            [
                [variable_pool.id(f"v_{i}_{j}") for j in range(num_nodes_g2)]
                for i in range(num_nodes_g1)
            ],
            dtype=int,
        )
        vid2mapping = {v: idx for idx, v in np.ndenumerate(variables)}

        cnf1, cnf2 = get_cnfs(num_nodes_g1, num_nodes_g2, program_interactions, swap_strategy, num_layers, variables)
        logger.info(f'Hard constraints: {len(cnf1)}')
        logger.info(f'Soft constraints: {len(cnf2)}')
        wcnf = WCNF()
        wcnf.extend(cnf1)
        wcnf.extend(cnf2, weights=[1]*len(cnf2))
        
        
        id = uuid.uuid1()
        wcnf.to_file(f'./{num_nodes_g1}.{num_nodes_g2}.{num_layers}.{id}.wcnf')
                
                
        time_limit=str(self.timeout)
        mem_limit=str(2**20)
        subprocess.run(
            ["/nfs/users/nfs_j/jc59/quantumwork/pangenome/sat/run", "--timestamp", "-d", "15", 
             "-o", f"{num_nodes_g1}.{num_nodes_g2}.{num_layers}.{id}.out", "-v", f"{num_nodes_g1}.{num_nodes_g2}.{num_layers}.{id}.var", 
             "-w", f"{num_nodes_g1}.{num_nodes_g2}.{num_layers}.{id}.wat", "-C", time_limit, "-W", time_limit, "-M" , mem_limit, 
                "/nfs/users/nfs_j/jc59/quantumwork/pangenome/sat/NuWLS-c_static", f'./{num_nodes_g1}.{num_nodes_g2}.{num_layers}.{id}.wcnf']
        )
        
        os.remove(f"{num_nodes_g1}.{num_nodes_g2}.{num_layers}.{id}.var")
        os.remove(f"{num_nodes_g1}.{num_nodes_g2}.{num_layers}.{id}.wat")
        os.remove(f"{num_nodes_g1}.{num_nodes_g2}.{num_layers}.{id}.wcnf")
        
        try:
            with open(f'{num_nodes_g1}.{num_nodes_g2}.{num_layers}.{id}.out', 'r') as f:
                out = f.readlines()
                
            os.remove(f"{num_nodes_g1}.{num_nodes_g2}.{num_layers}.{id}.out")
            out_data = [x.split() for x in out if len(x) > 0]
            sol, cost, satisfiable = None, None, None
            for line in out_data[::-1]:
                if line[1] == 'v':
                    sol = [int(c) for c in line[2]]
                elif line[1] == 'o':
                    cost = line[2]
                elif line[1] == 's':
                    satisfiable = True
                else: 
                    continue
                if satisfiable and sol is not None and cost is not None:
                    # for clause in cnf2:
                    #     if not np.any([
                    #         sol[np.abs(x) - 1] if np.sign(x) == 1 else 1 - sol[np.abs(x) - 1] for x in clause
                    #     ]):
                    #         logger.info(clause)
                    return {num_layers: (int(cost), [vid2mapping[idx+1] for idx in range(len(sol)) if sol[idx] > 0])}
                
        except Exception as e:
            logger.error('Could not parse SAT data')
            logger.error(e)
            return None
    


    def find_hubo_mappings(
        self,
        program_interactions: list[tuple],
        swap_strategy: ExtendedSwapStrategy,
        min_layers: int,
        max_layers: int
    ) -> dict[int, SATResult]:
        """Binary-search for the minimum swap depth satisfying all interaction constraints.

        Encodes the placement problem as a satisfiability instance and uses
        PySAT to perform a binary search over ``[min_layers, max_layers]``.
        The distance tensors for all required interaction orders are
        pre-populated in the swap strategy before the search begins.

        Args:
            program_interactions: List of interaction tuples to route.
            swap_strategy: The ``ExtendedSwapStrategy`` providing distance
                tensors; will be updated in-place with new distance tensors for
                all interaction orders found in ``program_interactions``.
            min_layers: Lower bound for the binary search (inclusive).
            max_layers: Upper bound for the binary search (inclusive).

        Returns:
            A dict mapping each tested ``num_layers`` value to a ``SATResult``
            (status, model, mapping, solve_time).  Only layers that were
            actually tested appear in the dict.
        """
        program_interactions = sorted(program_interactions,key=lambda e: len(e))
        nodes = set([node for interaction in program_interactions for node in interaction])
        num_nodes_g1 = len(nodes)
        num_nodes_g2: int = swap_strategy.distance_matrix.shape[0]
                
        if num_nodes_g1 > num_nodes_g2:
            return {num_nodes_g2: SATResult(False, [], [], 0)}
            
        if min_layers is None:
            min_layers = 0
        if max_layers is None:
            max_layers = num_nodes_g2 - 1

        variable_pool = IDPool(start_from=1)
        variables = np.array(
            [
                [variable_pool.id(f"v_{i}_{j}") for j in range(num_nodes_g2)]
                for i in range(num_nodes_g1)
            ],
            dtype=int,
        )
        vid2mapping = {v: idx for idx, v in np.ndenumerate(variables)}
        binary_search_results = {}

        def interrupt(solver: Solver):
            solver.interrupt()

        # Populate the distance tensors
        for l in set([len(interaction) for interaction in program_interactions if len(interaction) > 2]):
            logger.info(f'Start populating order {l} distance tensor')
            swap_strategy.distance_tensor(l)
            logger.info(f'Finished populating order {l} distance tensor')

        # Perform a binary search over the number of swap layers to find the minimum
        # number of swap layers that satisfies the subgraph isomorphism problem.
        while min_layers < max_layers:
            num_layers = (min_layers + max_layers) // 2
            logger.info(f'Num layers: {num_layers}')

            cnf1, cnf2 = get_cnfs(program_interactions, swap_strategy, num_layers, variables)
            cnf = CNF(from_clauses=cnf1 + cnf2)   

            with Solver(bootstrap_with=cnf, use_timer=True) as solver:
                # Solve the SAT problem with a timeout.
                # Timer is used to interrupt the solver when the timeout is reached.
                timer = Timer(self.timeout, interrupt, [solver])
                timer.start()
                status = solver.solve_limited(expect_interrupt=True)
                timer.cancel()
                # Get the solution and the elapsed time.
                sol = solver.get_model()
                e_time = solver.time()

                if status:
                    # If the SAT problem is satisfiable, convert the solution to a mapping.
                    mapping = [vid2mapping[idx] for idx in sol if idx > 0]
                    binary_search_results[num_layers] = SATResult(status, sol, mapping, e_time)
                    max_layers = num_layers
                else:
                    # If the SAT problem is unsatisfiable, return the last satisfiable solution.
                    binary_search_results[num_layers] = SATResult(status, sol, [], e_time)
                    min_layers = num_layers + 1

        return binary_search_results
        