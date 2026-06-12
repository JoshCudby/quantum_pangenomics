"""EstimatorV2 wrapper that records all circuit evaluations during optimisation.

Provides ``EstimatorWithHistory``, a drop-in replacement for Aer's
``EstimatorV2`` that saves the raw measurement counts alongside the expectation
value for every pub evaluated.  The extra data is exposed through
``PubResult.data.counts`` and is used for debugging optimisation convergence
and performing post-hoc solution analysis without re-running the circuit.
"""

from qiskit_aer.primitives import EstimatorV2
import numpy as np
from qiskit.primitives.containers import DataBin, PubResult, PrimitiveResult
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.quantum_info import Pauli
from qiskit_qaoa.utils.logging import get_logger

logger = get_logger(__name__)

class EstimatorWithHistory(EstimatorV2):
    """Aer EstimatorV2 subclass that attaches raw counts to every PubResult.

    During QAOA parameter optimisation the cost function is evaluated many
    times.  This class stores the measurement counts for every circuit
    execution so that:

    - Convergence can be diagnosed by inspecting the energy trajectory.
    - The best bitstrings seen across all iterations can be retrieved without
      re-running the circuit at the final parameters.

    The ``counts`` field is injected into the returned ``DataBin`` alongside
    the standard ``evs`` (expectation values) and ``stds`` arrays.
    """

    def _run(self, pubs: list[EstimatorPub]) -> PrimitiveResult[PubResult]:
        """Run a list of estimator pubs and collect history for each.

        Args:
            pubs: A list of ``EstimatorPub`` objects, each specifying a
                circuit, observables, parameter values, and precision.

        Returns:
            A ``PrimitiveResult`` containing one ``PubResult`` per pub with
            ``evs``, ``stds``, and ``counts`` populated in its ``DataBin``.
        """
        return PrimitiveResult([self._run_pub_with_history(pub) for pub in pubs], metadata={"version": 2})

    def _run_pub_with_history(self, pub: EstimatorPub) -> PubResult:
        """Run a single pub and attach measurement counts to the result.

        Saves expectation values via Aer's ``save_expectation_value`` snapshot
        instruction for each Pauli in the observable, then broadcasts parameter
        and observable shapes to produce the full ``evs`` array.  If
        ``precision > 0`` Gaussian noise is added to simulate shot noise.

        Args:
            pub: The ``EstimatorPub`` to evaluate.

        Returns:
            A ``PubResult`` whose ``DataBin`` contains:

            - ``evs``: Numpy array of expectation values with shape matching
              the broadcast of ``parameter_values`` and ``observables``.
            - ``stds``: Array filled with ``pub.precision``.
            - ``counts``: Raw measurement counts dict from the Aer backend.

        Raises:
            ValueError: If the operator is non-Hermitian and noise is
                requested (``precision > 0``).
        """
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