from qiskit_aer.primitives import EstimatorV2
import numpy as np
from qiskit.primitives.containers import DataBin, PubResult, PrimitiveResult
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.quantum_info import Pauli
from qiskit_qaoa.utils.logging import get_logger

logger = get_logger(__name__)

class EstimatorWithHistory(EstimatorV2):
    def _run(self, pubs: list[EstimatorPub]) -> PrimitiveResult[PubResult]:
        return PrimitiveResult([self._run_pub_with_history(pub) for pub in pubs], metadata={"version": 2})
    
    def _run_pub_with_history(self, pub: EstimatorPub) -> PubResult:
        circuit = pub.circuit.copy()
        observables = pub.observables
        parameter_values = pub.parameter_values
        precision = pub.precision

        # calculate broadcasting of parameters and observables
        param_shape = parameter_values.shape
        param_indices = np.fromiter(np.ndindex(param_shape), dtype=object).reshape(param_shape)
        bc_param_ind, bc_obs = np.broadcast_arrays(param_indices, observables)

        parameter_binds = {}
        param_array = parameter_values.as_array(circuit.parameters)
        parameter_binds = {p: param_array[..., i].ravel() for i, p in enumerate(circuit.parameters)}

        # save expval
        paulis = {pauli for obs_dict in observables.ravel() for pauli in obs_dict.keys()}
        for pauli in paulis:
            circuit.save_expectation_value(
                Pauli(pauli), qubits=range(circuit.num_qubits), label=pauli
            )
        result = self._backend.run(
            circuit, parameter_binds=[parameter_binds], **self.options.run_options
        ).result()

        # calculate expectation values (evs) and standard errors (stds)
        flat_indices = list(param_indices.ravel())
        evs = np.zeros_like(bc_param_ind, dtype=float)
        stds = np.full(bc_param_ind.shape, precision)
        for index in np.ndindex(*bc_param_ind.shape):
            param_index = bc_param_ind[index]
            flat_index = flat_indices.index(param_index)
            for pauli, coeff in bc_obs[index].items():
                expval = result.data(flat_index)[pauli]
                evs[index] += expval * coeff
        if precision > 0:
            rng = np.random.default_rng(self.options.run_options.get("seed_simulator"))
            if not np.all(np.isreal(evs)):
                raise ValueError("Given operator is not Hermitian and noise cannot be added.")
            evs = rng.normal(evs, precision, evs.shape)
        return PubResult(
            DataBin(evs=evs, stds=stds, counts=result.get_counts(), shape=evs.shape),
            metadata={
                "target_precision": precision,
                "circuit_metadata": pub.circuit.metadata,
                "simulator_metadata": result.metadata,
            },
        )