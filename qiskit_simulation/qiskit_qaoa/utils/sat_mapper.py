from __future__ import annotations

from itertools import combinations, product
from threading import Timer

import numpy as np

from pysat.formula import CNF, IDPool
from pysat.solvers import Solver

from qopt_best_practices.sat_mapping import SATMapper
from qopt_best_practices.sat_mapping.sat_mapper import SATResult

from qiskit_qaoa.utils.transpiler_passes import ExtendedSwapStrategy
from qiskit_qaoa.utils.logging import get_logger

logger = get_logger(__name__)

class HigherOrderSatMapper(SATMapper):
    def find_hubo_mappings(
        self, 
        program_interactions: list[tuple], 
        swap_strategy: ExtendedSwapStrategy, 
        min_layers: int, 
        max_layers: int 
    ) -> dict[int, SATResult]:
        program_interactions = sorted(program_interactions,key=lambda e: len(e))
        nodes = set([node for interaction in program_interactions for node in interaction])
        num_nodes_g1 = len(nodes)
        num_nodes_g2: int = swap_strategy.distance_matrix.shape[0]
        
        logger.info(num_nodes_g1, num_nodes_g2)
        
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

        # Perform a binary search over the number of swap layers to find the minimum
        # number of swap layers that satisfies the subgraph isomorphism problem.
        while min_layers < max_layers:
            num_layers = (min_layers + max_layers) // 2
            logger.info(f'Num layers: {num_layers}')

            # Make a cnf for the adjacency constraint
            cnf2 = []
            
            old_num_qubits = 0
            distance_tensor = np.array(0) 
            for interaction in program_interactions:
                num_qubits = len(interaction)
                if num_qubits != old_num_qubits:
                    old_num_qubits = num_qubits
                    if num_qubits == 2:
                        distance_tensor = swap_strategy.distance_matrix
                    else:
                        logger.info(f'Start populating order {num_qubits} distance tensor')
                        distance_tensor = swap_strategy.distance_tensor(num_qubits)
                        logger.info(f'Finished populating order {num_qubits} distance tensor')
                    
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
        