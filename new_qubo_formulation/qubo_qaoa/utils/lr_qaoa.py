import numpy as np
from typing import Optional

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import QAOAAnsatz, PauliEvolutionGate
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_ibm_runtime import IBMBackend

from qopt_best_practices.transpilation import qaoa_swap_strategy_pm

from qubo_qaoa.utils.swap_strategy import QUBOSwapStrategy

from qiskit_qaoa.utils.logging import get_logger
logger = get_logger(__name__)


def print_circuit_info(qc, circuit_name):
    logger.info(f'{circuit_name} has {qc.count_ops().get("cz", 0) + qc.count_ops().get("rzz", 0) + qc.count_ops().get("cx", 0) + qc.count_ops().get("ecr", 0)} 2Q gates \
    and {qc.depth(lambda instr: len(instr.qubits) > 1)} 2Q depth')
    logger.info(qc.count_ops())


def _init_layer(num_qubits, phis: Optional[ParameterVector]):
    init = QuantumCircuit(num_qubits, num_qubits)
    if phis is not None:
        if not len(phis) == num_qubits:
            raise Exception(f'Wrong number of phis, expected {num_qubits}, got {len(phis)}')
        for i in range(num_qubits):
            init.ry(phis[i], i)
    else:
        init.h(range(num_qubits))  
    return init
        

def _mixer_layer(num_qubits, phis: Optional[ParameterVector]):
    mixer = QuantumCircuit(num_qubits)
    beta = Parameter("β")
    if phis is not None:
        for i in range(num_qubits):
            mixer.ry(-phis[i], i)
            mixer.rz(-2 * beta, i)
            mixer.ry(phis[i], i)
    else:
        mixer.rx(-2 * beta, range(num_qubits))
    return mixer


def _hardware_circuit_construction(
    num_virtual_qubits: int,
    cost_op: SparsePauliOp,
    sat_map: dict,
    p: int,
    backend: Optional[IBMBackend],
    edge_colouring,
    swap_strategy: QUBOSwapStrategy,
    phis: Optional[ParameterVector]
):
    n = swap_strategy._num_vertices

    singles = cost_op[cost_op.paulis.z.sum(axis=-1) == 1]
    doubles = cost_op[cost_op.paulis.z.sum(axis=-1) == 2]
    doubles_circ = QAOAAnsatz(
        doubles,
        initial_state=QuantumCircuit(n),
        mixer_operator=QuantumCircuit(n)
    )
    config = {
        "num_layers": 1,
        "swap_strategy": swap_strategy,
        "edge_coloring": edge_colouring,
        "construct_qaoa": False,
        "basis_gates": ["rz", "cx", "swap"]
    }
    
    properties = {}
    def get_permutation(pass_, dag, time, property_set, count):
        properties["virtual_permutation_layout"] = property_set["virtual_permutation_layout"]
    pm = qaoa_swap_strategy_pm(config)
    tdoubles_circ: QuantumCircuit = pm.run(doubles_circ, callback=get_permutation)
    
    singles_circ = QuantumCircuit(n)
    singles_circ.append(PauliEvolutionGate(singles, time=tdoubles_circ.parameters[0]), range(n))
    tsingles: QuantumCircuit = transpile(singles_circ, basis_gates=["rz"])
    cost_circ: QuantumCircuit = tsingles.compose(tdoubles_circ, inplace=False)
    
    init_state = QuantumCircuit(n)
    if phis is not None:
        if not len(phis) == num_virtual_qubits:
            raise Exception(f'Wrong number of phis, expected {num_virtual_qubits}, got {len(phis)}')
        for i in range(num_virtual_qubits):
            qubit = sat_map[i]
            init_state.ry(phis[i], qubit)
    else:
        init_state.h([x for x in sat_map.values()])  
        
        
    mixer_layer_even = QuantumCircuit(n)
    beta = Parameter("β")
    if phis is not None:
        inv_sat_map = {v: k for k, v in sat_map.items()}
        for i, qidx in [(inv_sat_map[x], properties['virtual_permutation_layout'].get_physical_bits()[x]) for x in sat_map.values()]:
            mixer_layer_even.ry(-phis[i], qidx)
            mixer_layer_even.rz(-2 * beta, qidx)
            mixer_layer_even.ry(phis[i], qidx)
    else:
        mixer_layer_even.rx(-2 * beta, [properties['virtual_permutation_layout'].get_physical_bits()[x] for x in sat_map.values()])
    
    
    mixer_layer_odd = QuantumCircuit(n)
    if phis is not None:
        for i in range(num_virtual_qubits):
            qubit = sat_map[i]
            mixer_layer_odd.ry(-phis[i], qubit)
            mixer_layer_odd.rz(-2 * beta, qubit)
            mixer_layer_odd.ry(phis[i], qubit)
    else:
        mixer_layer_odd.rx(-2 * beta, [x for x in sat_map.values()])
    
    gammas = ParameterVector("γ",p)
    betas = ParameterVector("β", p)

    qaoa_circuit = QuantumCircuit(n, num_virtual_qubits)

    qaoa_circuit.compose(init_state, inplace=True)
    for layer in range(p):
        bind_dict = {cost_circ.parameters[0]: gammas[layer]}
        bound_cost_layer = cost_circ.assign_parameters(bind_dict)

        mixer_layer = mixer_layer_even if layer % 2 == 0 else mixer_layer_odd
        bind_dict = {mixer_layer.parameters[0]: betas[layer]}
        bound_mixer_layer = mixer_layer.assign_parameters(bind_dict)

        if layer % 2 == 0:
            # even layer -> append cost
            qaoa_circuit.compose(bound_cost_layer, range(n), inplace=True)
        else:
            # odd layer -> append reversed cost
            qaoa_circuit.compose(
                bound_cost_layer.reverse_ops(), range(n), inplace=True
            )
        qaoa_circuit.compose(bound_mixer_layer, range(n), inplace=True)

    if p % 2 == 1:
        # iterate over layout permutations to recover measurements
        if properties["virtual_permutation_layout"]:
            inv_sat_map = {v: k for k, v in sat_map.items()}
            for cidx, qidx in [(inv_sat_map[x], properties['virtual_permutation_layout'].get_physical_bits()[x]) for x in sat_map.values()]:
                qaoa_circuit.measure(qidx, cidx)
        else:
            print("layout not found, assigining trivial layout")
            for cidx, qidx in sat_map.items():
                qaoa_circuit.measure(qidx, cidx)
    else:
        for cidx, qidx in sat_map.items():
            qaoa_circuit.measure(qidx, cidx)
            

    generic_pm = generate_preset_pass_manager(optimization_level=3, backend=backend) # , scheduling_method="asap"
    backend_circ = generic_pm.run(qaoa_circuit)
        
    return backend_circ


def get_LR_qaoa_circuit(
    p: int, 
    delta_b: float, 
    delta_g: float, 
    num_qubits: int,
    hamiltonian: SparsePauliOp,
    qaoa_circ: Optional[QuantumCircuit],
    phis: Optional[ParameterVector],
    measure: Optional[bool]
) -> tuple[QuantumCircuit, QuantumCircuit]:
    x = np.array([(j-0.5)/p for j in range(1, p+1)])
    betas = delta_b * (1-x)
    gammas = delta_g * x
    fixed_params = list(betas) + list(gammas)
    
    if qaoa_circ is None:
        init = _init_layer(num_qubits, phis)
        mixer = _mixer_layer(num_qubits, phis)
                    
        gamma = Parameter("γ")
        cost_circuit = QuantumCircuit(num_qubits)
        cost_circuit.append(PauliEvolutionGate(hamiltonian, time=gamma), range(num_qubits))
        cost_circuit = transpile(cost_circuit)
        
        gamma_params = ParameterVector("γ", p)
        beta_params = ParameterVector("β", p)
        
        circuit = QuantumCircuit(num_qubits, num_qubits)
        circuit.compose(init, range(num_qubits), inplace=True)
        for layer in range(p):
            bind_dict = {cost_circuit.parameters[0]: gamma_params[layer]}
            bound_cost_layer = cost_circuit.assign_parameters(bind_dict)

            bind_dict = {mixer.parameters[0]: beta_params[layer]}
            bound_mixer_layer = mixer.assign_parameters(bind_dict)
            
            circuit.compose(bound_cost_layer, range(num_qubits), inplace=True)
            circuit.compose(bound_mixer_layer, range(num_qubits), inplace=True)
        

        if measure:
            circuit.measure_all(add_bits=False)
        else:
            circuit.save_statevector()
        
        logger.info(f'p = {p}. Circuit depth: {circuit.depth()}')
    else:
        circuit = qaoa_circ
        
    fixed_param_bind = {circuit.parameters[i]: fixed_params[i] for i in range(2*p)}
    fixed_qc = circuit.assign_parameters(fixed_param_bind)
    return fixed_qc, circuit


def get_hardware_LR_qaoa_circuit(
    p: int, 
    delta_b: float, 
    delta_g: float, 
    num_virtual_qubits: int,
    remapped_cost_op: SparsePauliOp,
    sat_map: dict[int, int],
    backend: IBMBackend,
    edge_colouring: dict[tuple[int, int], int],
    swap_strategy: QUBOSwapStrategy,
    qaoa_circ: Optional[QuantumCircuit],
    phis: Optional[ParameterVector],
) -> tuple[QuantumCircuit, QuantumCircuit]:
    x = np.array([(j-0.5)/p for j in range(1, p+1)])
    betas = delta_b * (1-x)
    gammas = delta_g * x
    fixed_params = list(betas) + list(gammas)
    
    if qaoa_circ is None:
        circuit = _hardware_circuit_construction(
            num_virtual_qubits, remapped_cost_op, sat_map, p, backend, edge_colouring, swap_strategy, phis
        )
        
    else:
        circuit = qaoa_circ
        
    fixed_param_bind = {circuit.parameters[i]: fixed_params[i] for i in range(2*p)}
    fixed_qc = circuit.assign_parameters(fixed_param_bind)
    return fixed_qc, circuit