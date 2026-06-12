"""Qiskit TransformationPass for QAOA circuit construction.

Provides ``QAOAPass``, a ``TransformationPass`` that wraps a compiled cost
layer DAG and assembles it into a complete QAOA circuit by prepending an
initial state, interleaving alternating (forward / reversed) cost layers with
mixer layers, and appending layout-aware measurements.
"""

from qiskit import QuantumCircuit

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit import ParameterVector

class QAOAPass(TransformationPass):
    """Assemble a QAOA circuit from a pre-compiled cost-layer DAG.

    Accepts the cost-layer DAG (already compiled with a swap strategy or
    other routing) and wraps it with an initial state and mixer layers to
    produce a full p-layer QAOA ansatz.  Even-indexed layers use the cost
    circuit in its original gate order; odd-indexed layers use it in reversed
    gate order to reduce depth via cancellations across layer boundaries.

    The pass reads ``property_set["virtual_permutation_layout"]`` (set by
    earlier routing passes) to assign correct measurements when the number of
    layers is odd (in which case the qubit permutation is non-trivial).
    """

    def __init__(self, num_layers, num_qubits, init_state=None, mixer_layer=None, betas: ParameterVector = None):
        """Initialise the QAOA pass.

        Args:
            num_layers: Number of QAOA layers ``p``.
            num_qubits: Total number of qubits in the circuit.
            init_state: Optional ``QuantumCircuit`` for the initial state.
                Defaults to an equal superposition (Hadamard on all qubits).
            mixer_layer: Optional pre-built mixer ``QuantumCircuit``.  If
                ``None``, the default RX-rotation mixer is constructed from
                ``betas``.
            betas: A ``ParameterVector`` used to construct the default mixer.
                Must be provided when ``mixer_layer`` is ``None``.

        Raises:
            Exception: If neither ``mixer_layer`` nor ``betas`` is provided.
        """
        if mixer_layer is None and betas is None:
            raise Exception('Either mixer or betas should be provided')
        super().__init__()
        self.num_layers = num_layers
        self.num_qubits = num_qubits
        
        if init_state is None:
            # Add default initial state -> equal superposition
            self.init_state = QuantumCircuit(num_qubits)
            self.init_state.h(range(num_qubits))
        else: 
            self.init_state = init_state
        
        if mixer_layer is None:
            # Define default mixer layer
            self.mixer_layer = QuantumCircuit(num_qubits)
            self.mixer_layer.rx(-2*betas[0], range(num_qubits))
        else:
            self.mixer_layer = mixer_layer

    def run(self, cost_layer_dag):
        """Assemble the full QAOA circuit from the cost-layer DAG.

        Re-parametrises the cost layer with fresh ``ParameterVector`` symbols
        ``γ`` and ``β``, then builds the QAOA circuit by composing:
        ``init_state → (cost_layer → mixer_layer) * num_layers``.
        Odd layers use the reversed cost circuit.  Measurements are appended
        using the virtual permutation layout when ``num_layers`` is odd.

        Args:
            cost_layer_dag: A ``DAGCircuit`` for a single compiled QAOA cost
                layer (one parameter controlling all gates).

        Returns:
            A ``DAGCircuit`` representing the complete QAOA circuit with
            measurements.
        """

        cost_layer = dag_to_circuit(cost_layer_dag)
        qaoa_circuit = QuantumCircuit(self.num_qubits, self.num_qubits)
        # Re-parametrize the circuit
        gammas = ParameterVector("γ", self.num_layers)
        betas = ParameterVector("β", self.num_layers)

        # Add initial state
        qaoa_circuit.compose(self.init_state, inplace = True)

        # iterate over number of qaoa layers
        # and alternate cost/reversed cost and mixer
        for layer in range(self.num_layers): 
        
            bind_dict = {cost_layer.parameters[0]: gammas[layer]}
            bound_cost_layer = cost_layer.assign_parameters(bind_dict)
            
            bind_dict = {self.mixer_layer.parameters[0]: betas[layer]}
            bound_mixer_layer = self.mixer_layer.assign_parameters(bind_dict)
        
            if layer % 2 == 0:
                # even layer -> append cost
                qaoa_circuit.compose(bound_cost_layer, range(self.num_qubits), inplace=True)
            else:
                # odd layer -> append reversed cost
                qaoa_circuit.compose(bound_cost_layer.reverse_ops(), range(self.num_qubits), inplace=True)
        
            # the mixer layer is not reversed
            qaoa_circuit.compose(bound_mixer_layer, range(self.num_qubits), inplace=True)
        

        if self.num_layers % 2 == 1:
            # iterate over layout permutations to recover measurements
            if self.property_set["virtual_permutation_layout"]:
                for cidx, qidx in self.property_set["virtual_permutation_layout"].get_physical_bits().items():
                    qaoa_circuit.measure(qidx, cidx)
            else:
                print("layout not found, assigining trivial layout")
                for idx in range(self.num_qubits):
                    qaoa_circuit.measure(idx, idx)
        else:
            for idx in range(self.num_qubits):
                qaoa_circuit.measure(idx, idx)
    
        return circuit_to_dag(qaoa_circuit)