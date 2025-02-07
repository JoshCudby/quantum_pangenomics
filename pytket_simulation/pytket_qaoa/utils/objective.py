from datetime import datetime
from pytket import Circuit
from pytket.utils.operators import QubitPauliOperator
from pytket.partition import measurement_reduction, MeasurementSetup, PauliPartitionStrat, MeasurementBitMap
from pytket.backends.backendresult import BackendResult

import qnexus as qnx
import numpy as np

class Objective:
    def __init__(
        self,
        symbolic_circuit: qnx.circuits.CircuitRef,
        problem_hamiltonian: QubitPauliOperator,
        n_shots_per_circuit: int,
        target: qnx.BackendConfig,
        iteration_number: int = 0,
        n_iterations: int = 10,
    ) -> None:
        """Returns the objective function needed for a variational
        procedure.
        """
        terms = [term for term in problem_hamiltonian._dict.keys()]
        self._symbolic_circuit: Circuit = symbolic_circuit.download_circuit()
        self._hamiltonian: QubitPauliOperator = problem_hamiltonian
        self._nshots: int = n_shots_per_circuit
        self._measurement_setup: MeasurementSetup = measurement_reduction(
            terms, strat=PauliPartitionStrat.CommutingSets
        )
        self._iteration_number: int = iteration_number
        self._niters: int = n_iterations
        self._target = target


    def __call__(self, parameters: np.ndarray) -> float:
        value = self._objective_function(parameters)
        self._iteration_number += 1
        if self._iteration_number >= self._niters:
            self._iteration_number = 0
        return value
    
    
    def _compute_expectation_paulistring(
        self,
        distribution: dict[tuple[int, ...], float], 
        bitmap: MeasurementBitMap
    ) -> float:
        value = 0
        for bitstring, probability in distribution.items():
            value += probability * (sum(bitstring[i] for i in bitmap.bits) % 2)
        return ((-1) ** bitmap.invert) * (-2 * value + 1)
    
    
    def _compute_expectation_value(
        self,
        results: list[BackendResult],
        measurement_setup: MeasurementSetup,
        operator: QubitPauliOperator,
    ) -> float:
        energy = 0
        for pauli_string, bitmaps in measurement_setup.results.items():
            string_coeff = operator.get(pauli_string, 0.0)
            if string_coeff > 0:
                for bm in bitmaps:
                    index = bm.circ_index
                    distribution = results[index].get_distribution()
                    value = self._compute_expectation_paulistring(distribution, bm)
                    energy += complex(value * string_coeff).real
        return energy
    
    
    def _sort_free_symbols(self, symbol_set: set):
        return sorted(symbol_set, key=lambda x: x.name)
    
    
    def _objective_function(
        self,
        parameters: np.ndarray,
    ) -> float:
        
        # Prepare the parameterised state preparation circuit
        assert len(parameters) == len(self._symbolic_circuit.free_symbols())
        symbol_dict = {s: p for s, p in zip(self._sort_free_symbols(self._symbolic_circuit.free_symbols()), parameters)}
        state_prep_circuit = self._symbolic_circuit.copy()
        state_prep_circuit.symbol_substitution(symbol_dict)

        # Label each job with the properties associated with the circuit.
        properties = {str(sym): val for sym, val in symbol_dict.items()} | {"iteration": self._iteration_number}

        with qnx.context.using_properties(**properties):

            circuit_list = self._build_circuits(state_prep_circuit)

            # Execute circuits with Nexus
            results = qnx.execute(
                name=f"execute_job_QAOA_{datetime.now()}_{self._iteration_number}",
                circuits=circuit_list,
                n_shots=[self._nshots]*len(circuit_list),
                backend_config=self._target,
                timeout=None,
            )

        expval = self._compute_expectation_value(
            results, self._measurement_setup, self._hamiltonian
        )
        return expval


    def _build_circuits(self, state_prep_circuit: Circuit) -> list[qnx.circuits.CircuitRef]:
        # Requires properties to be set in the context
        
        # Upload the numerical state-prep circuit to Nexus
        qnx.circuits.upload(
            circuit=state_prep_circuit,
            name=f"state prep circuit {self._iteration_number}",
        )
        circuit_list = []
        for mc in self._measurement_setup.measurement_circs:
            c = state_prep_circuit.copy()
            c.append(mc)
            # Upload each measurement circuit to Nexus with correct params
            measurement_circuit_ref = qnx.circuits.upload(
                circuit=c, 
                name=f"state prep circuit {self._iteration_number}",
            )
            circuit_list.append(measurement_circuit_ref)

        # Compile circuits with Nexus
        compiled_circuit_refs = qnx.compile(
            name=f"compile_job_QAOA_{datetime.now()}_{self._iteration_number}",
            circuits=circuit_list,
            optimisation_level=2,
            backend_config=self._target,
            timeout=None,
        )
            
        return compiled_circuit_refs