import unittest
from qiskit import QuantumCircuit, transpile, QuantumRegister
from qiskit.circuit import Instruction

import numpy as np
import networkx as nx

from qiskit_aer import AerSimulator
from qiskit_prog_qaoa.utils.oriented_circuit_utils import (
    is_equal_to_with_parity, compute_next_nodes_with_parity,
    get_phase_operator_gate, get_constraint_circuit, get_objective_circuit
)

backend_options = dict(
    method='statevector',
    device='GPU',
    cuStateVec_enable=True,
    precision='single'
)


def toy_graph():
    g = nx.DiGraph()
    g.add_nodes_from([
        ('u0_+', {"weight": 2}),
        ('u0_-', {"weight": 2}),
        ('u1_+', {"weight": 1}),
        ('u1_-', {"weight": 1}),
        ('u2_+', {"weight": 1}),
        ('u2_-', {"weight": 1}),
    ])
    g.add_edges_from([
        ('u0_+','u1_+'), ('u1_+','u2_-'), ('u2_-', 'u0_+'),
        ('u1_-','u0_-'), ('u2_+','u1_-'), ('u0_-', 'u2_+'),
    ])
    return g


def get_single_bitstring_mapping_with_amplitudes(
    input_str: str,
    circuit_to_run: QuantumCircuit | Instruction,
    simulator: AerSimulator,
    parameters: list | None
):
    circ = QuantumCircuit(len(input_str))
    for idx, char in enumerate(input_str):
        if char == '1':
            circ.x(idx)
    circ.save_statevector('start')

    circ.compose(
        circuit_to_run,
        list(range(circuit_to_run.num_qubits)),
        inplace=True
    )
    circ.save_statevector('end')
    
    if parameters is not None:
        circ.assign_parameters(parameters, inplace=True)
        
    t_circ = transpile(circ, simulator, optimization_level=3)
    result = simulator.run(t_circ).result()
    start_data = result.data()['start'].data
    start_ket = np.binary_repr(np.nonzero(start_data)[0][0], len(input_str))[::-1]
    start_amplitude = start_data[np.nonzero(start_data)[0][0]]

    end_data = result.data()['end'].data
    if len(np.nonzero(end_data)) > 1:
        raise Exception('Mapped to several non-zero amplitudes')
    end_ket = np.binary_repr(np.nonzero(end_data)[0][0], len(input_str))[::-1]
    end_amplitude = end_data[np.nonzero(end_data)[0][0]]

    return start_ket, start_amplitude, end_ket, end_amplitude


class TestIsEqualToWithParityCircuit(unittest.TestCase):
    def setUp(self):
        self.simulator = AerSimulator(**backend_options)
        self.n = 3
        self.j = 2
        self.b = 0

        self.qc = is_equal_to_with_parity(self.n, self.j, self.b)


    def test_is_equal(self):
        input_str = '00100'
        
        start_ket, start_amplitude, end_ket, end_amplitude = get_single_bitstring_mapping_with_amplitudes(
            input_str,
            self.qc,
            self.simulator,
            None
        )
        self.assertEqual(
            start_ket,
            input_str
        )
        self.assertAlmostEqual(
            start_amplitude,
            1. + 0.j
        )
        self.assertEqual(
            end_ket,
            '00101'
        )
        self.assertAlmostEqual(
            end_amplitude,
            1. + 0.j
        )
        
    def test_is_not_equal(self):
        input_str = '01000'
        
        start_ket, start_amplitude, end_ket, end_amplitude = get_single_bitstring_mapping_with_amplitudes(
            input_str,
            self.qc,
            self.simulator,
            None
        )
        self.assertEqual(
            start_ket,
            input_str
        )
        self.assertAlmostEqual(
            start_amplitude,
            1. + 0.j
        )
        self.assertEqual(
            end_ket,
            input_str
        )
        self.assertAlmostEqual(
            end_amplitude,
            1. + 0.j
        )


class TestOrientedObjectiveCircuit(unittest.TestCase):
    def setUp(self):
        self.simulator = AerSimulator(**backend_options)
        self.n = 3
        self.T = 4
        self.K = 2
        self.parameter = np.pi/16
        self.lamda = 3
        self.graph = toy_graph()

        self.qc = get_objective_circuit(self.n, self.K, self.T, self.graph, self.parameter, None)


    def test_optimal_path(self):
        optimal_path = '0100' + '0010' + '1110' + '0100'
        input_str =  optimal_path + '0' * (self.K * int(1 + np.ceil(np.log2(self.n+2))) + 2)
        start_ket, start_amplitude, end_ket, end_amplitude = get_single_bitstring_mapping_with_amplitudes(
            input_str,
            self.qc,
            self.simulator,
            None
        )
        self.assertEqual(
            start_ket,
            input_str
        )
        self.assertAlmostEqual(
            start_amplitude,
            1. + 0.j
        )
        self.assertEqual(
            end_ket,
            input_str
        )
        self.assertAlmostEqual(
            end_amplitude,
            1. + 0.j
        )
        

    def test_one_missing_weight(self):
        optimal_path = '0100' + '0010' + '1110' + '0001'
        input_str =  optimal_path + '0' * (self.K * int(1 + np.ceil(np.log2(self.n+2))) + 2)
        start_ket, start_amplitude, end_ket, end_amplitude = get_single_bitstring_mapping_with_amplitudes(
            input_str,
            self.qc,
            self.simulator,
            None
        )
        self.assertEqual(
            start_ket,
            input_str
        )
        self.assertAlmostEqual(
            start_amplitude,
            1. + 0.j
        )
        self.assertEqual(
            end_ket,
            input_str
        )
        self.assertAlmostEqual(
            end_amplitude,
            np.e ** (1j * self.parameter)
        )
        

    def test_missing_weight_two(self):
        optimal_path = '0010' + '1110' + '0001' + '0001'
        input_str =  optimal_path + '0' * (self.K * int(1 + np.ceil(np.log2(self.n+2))) + 2)
        start_ket, start_amplitude, end_ket, end_amplitude = get_single_bitstring_mapping_with_amplitudes(
            input_str,
            self.qc,
            self.simulator,
            None
        )
        self.assertEqual(
            start_ket,
            input_str
        )
        self.assertAlmostEqual(
            start_amplitude,
            1. + 0.j
        )
        self.assertEqual(
            end_ket,
            input_str
        )
        self.assertAlmostEqual(
            end_amplitude,
            np.e ** (4j * self.parameter)
        )
        
        
    def test_missing_two_weight_ones(self):
        optimal_path = '0100' + '0010' + '0001' + '0001'
        input_str =  optimal_path + '0' * (self.K * int(1 + np.ceil(np.log2(self.n+2))) + 2)
        start_ket, start_amplitude, end_ket, end_amplitude = get_single_bitstring_mapping_with_amplitudes(
            input_str,
            self.qc,
            self.simulator,
            None
        )
        self.assertEqual(
            start_ket,
            input_str
        )
        self.assertAlmostEqual(
            start_amplitude,
            1. + 0.j
        )
        self.assertEqual(
            end_ket,
            input_str
        )
        self.assertAlmostEqual(
            end_amplitude,
            np.e ** (2j * self.parameter),
            places=5
        )
        
        
class TestComputeNextNodesWithParityCircuit(unittest.TestCase):
    def setUp(self):
        self.simulator = AerSimulator(**backend_options)
        self.n = 3
        self.T = 3
        self.K = 2
        self.j = 1
        self.b = 0
        self.parameter = np.pi/16

        ceil_log_n2 = int(np.ceil(np.log2(self.n+2)))
        registers = {f'solution_{t}' : QuantumRegister(ceil_log_n2+1, name=f'solution_{t}') for t in range(self.T)}
        registers.update({f'next_node_{k}': QuantumRegister(ceil_log_n2+1, name=f'next_node_{k}') for k in range(self.K)})
        registers.update({'flag': QuantumRegister(1, name='flag')})
        registers.update({'visits_flag': QuantumRegister(1, name='visits_flag')})

        circuit = QuantumCircuit()
        for register in registers.values():
            circuit.add_register(register)
        self.qc = compute_next_nodes_with_parity(circuit, registers, self.j, self.b, self.n, self.K, self.T, self.parameter)


    def test_no_copy(self):
        path = '0010' + '0110' + '0110'
        input_str =  path + '0' * (self.K * int(1 + np.ceil(np.log2(self.n+2))) + 2)
        start_ket, start_amplitude, end_ket, end_amplitude = get_single_bitstring_mapping_with_amplitudes(
            input_str,
            self.qc,
            self.simulator,
            None
        )
        self.assertEqual(
            start_ket,
            input_str
        )
        self.assertAlmostEqual(
            start_amplitude,
            1. + 0.j
        )
        self.assertEqual(
            end_ket,
            input_str
        )
        self.assertAlmostEqual(
            end_amplitude,
            1. + 0.j
        )


    def test_copy_one(self):
        path = '0100' + '0110' + '0110'
        input_str =  path + '0' * (self.K * int(1 + np.ceil(np.log2(self.n+2))) + 2)
        start_ket, start_amplitude, end_ket, end_amplitude = get_single_bitstring_mapping_with_amplitudes(
            input_str,
            self.qc,
            self.simulator,
            None
        )
        self.assertEqual(
            start_ket,
            input_str
        )
        self.assertAlmostEqual(
            start_amplitude,
            1. + 0.j
        )
        self.assertEqual(
            end_ket,
            path + '0110' + '0000' + '00'
        )
        self.assertAlmostEqual(
            end_amplitude,
            1. + 0.j,
            places=5
        )
        
        
    def test_copy_two(self):
        path = '0100' + '0100' + '0110'
        input_str =  path + '0' * (self.K * int(1 + np.ceil(np.log2(self.n+2))) + 2)
        start_ket, start_amplitude, end_ket, end_amplitude = get_single_bitstring_mapping_with_amplitudes(
            input_str,
            self.qc,
            self.simulator,
            None
        )
        self.assertEqual(
            start_ket,
            input_str
        )
        self.assertAlmostEqual(
            start_amplitude,
            1. + 0.j
        )
        self.assertEqual(
            end_ket,
            path +  '0110' + '0100' + '00'
        )
        self.assertAlmostEqual(
            end_amplitude,
            1. + 0.j,
            places=5
        )
        

class TestOrientedConstraintCircuit(unittest.TestCase):
    def setUp(self):
        self.simulator = AerSimulator(**backend_options)
        self.n = 3
        self.T = 4
        self.K = 1
        self.parameter = np.pi/16
        self.lamda = 3
        self.graph = toy_graph()

        self.qc = get_constraint_circuit(self.n, self.K, self.T, self.graph, self.parameter, None)


    def test_optimal_path(self):
        optimal_path = '0100' + '0010' + '1110' + '0100'
        input_str =  optimal_path + '0' * (self.K * int(1 + np.ceil(np.log2(self.n+2))) + 2)
        start_ket, start_amplitude, end_ket, end_amplitude = get_single_bitstring_mapping_with_amplitudes(
            input_str,
            self.qc,
            self.simulator,
            None
        )
        self.assertEqual(
            start_ket,
            input_str
        )
        self.assertAlmostEqual(
            start_amplitude,
            1. + 0.j
        )
        self.assertEqual(
            end_ket,
            input_str
        )
        self.assertAlmostEqual(
            end_amplitude,
            1. + 0.j
        )
        
    def test_one_bad_step(self):
        optimal_path = '0100' + '0010' + '1110' + '1100'
        input_str =  optimal_path + '0' * (self.K * int(1 + np.ceil(np.log2(self.n+2))) + 2)
        start_ket, start_amplitude, end_ket, end_amplitude = get_single_bitstring_mapping_with_amplitudes(
            input_str,
            self.qc,
            self.simulator,
            None
        )
        self.assertEqual(
            start_ket,
            input_str
        )
        self.assertAlmostEqual(
            start_amplitude,
            1. + 0.j
        )
        self.assertEqual(
            end_ket,
            input_str
        )
        self.assertAlmostEqual(
            end_amplitude,
            np.e**(1j* self.parameter)
        )
        

    def test_two_bad_steps(self):
        optimal_path = '0100' + '1010' + '1110' + '0001'
        input_str =  optimal_path + '0' * (self.K * int(1 + np.ceil(np.log2(self.n+2))) + 2)
        start_ket, start_amplitude, end_ket, end_amplitude = get_single_bitstring_mapping_with_amplitudes(
            input_str,
            self.qc,
            self.simulator,
            None
        )
        self.assertEqual(
            start_ket,
            input_str
        )
        self.assertAlmostEqual(
            start_amplitude,
            1. + 0.j
        )
        self.assertEqual(
            end_ket,
            input_str
        )
        self.assertAlmostEqual(
            end_amplitude,
            np.e**(2j* self.parameter),
            places=5
        )
        
        
    def test_three_bad_steps(self):
        optimal_path = '0100' + '1010' + '1110' + '1100'
        input_str =  optimal_path + '0' * (self.K * int(1 + np.ceil(np.log2(self.n+2))) + 2)
        start_ket, start_amplitude, end_ket, end_amplitude = get_single_bitstring_mapping_with_amplitudes(
            input_str,
            self.qc,
            self.simulator,
            None
        )
        self.assertEqual(
            start_ket,
            input_str
        )
        self.assertAlmostEqual(
            start_amplitude,
            1. + 0.j
        )
        self.assertEqual(
            end_ket,
            input_str
        )
        self.assertAlmostEqual(
            end_amplitude,
            np.e**(3j* self.parameter),
            places=5
        )
   

class TestOrientedPhaseGate(unittest.TestCase):
    def setUp(self):
        self.simulator = AerSimulator(**backend_options)
        self.n = 3
        self.T = 4
        self.K = 2
        self.parameter = np.pi/16
        self.lamda = 3
        self.graph = toy_graph()

        self.qc = get_phase_operator_gate(self.n, self.K, self.T, self.graph, self.lamda, 0)


    def test_optimal_path(self):
        optimal_path = '0100' + '0010' + '1110' + '0100'
        input_str =  optimal_path + '0' * (self.K * int(1 + np.ceil(np.log2(self.n+2))) + 2)
        start_ket, start_amplitude, end_ket, end_amplitude = get_single_bitstring_mapping_with_amplitudes(
            input_str,
            self.qc,
            self.simulator,
            [self.parameter]
        )
        self.assertEqual(
            start_ket,
            input_str
        )
        self.assertAlmostEqual(
            start_amplitude,
            1. + 0.j
        )
        self.assertEqual(
            end_ket,
            input_str
        )
        self.assertAlmostEqual(
            end_amplitude,
            1. + 0.j,
            places=5
        )
        
        
    def test_missing_weight_path(self):
        optimal_path = '0100' + '0010' + '1110' + '0001'
        input_str =  optimal_path + '0' * (self.K * int(1 + np.ceil(np.log2(self.n+2))) + 2)
        start_ket, start_amplitude, end_ket, end_amplitude = get_single_bitstring_mapping_with_amplitudes(
            input_str,
            self.qc,
            self.simulator,
            [self.parameter]
        )
        self.assertEqual(
            start_ket,
            input_str
        )
        self.assertAlmostEqual(
            start_amplitude,
            1. + 0.j
        )
        self.assertEqual(
            end_ket,
            input_str
        )
        self.assertAlmostEqual(
            end_amplitude,
            np.e ** (1j * self.parameter),
            places=5
        )
        

    def test_wrong_order_path(self):
        optimal_path = '0010' + '1110' + '0100' + '0100'
        input_str =  optimal_path + '0' * (self.K * int(1 + np.ceil(np.log2(self.n+2))) + 2)
        start_ket, start_amplitude, end_ket, end_amplitude = get_single_bitstring_mapping_with_amplitudes(
            input_str,
            self.qc,
            self.simulator,
            [self.parameter]
        )
        self.assertEqual(
            start_ket,
            input_str
        )
        self.assertAlmostEqual(
            start_amplitude,
            1. + 0.j
        )
        self.assertEqual(
            end_ket,
            input_str
        )
        self.assertAlmostEqual(
            end_amplitude,
            np.e ** (1j * self.lamda * self.parameter),
            places=5
        )


if __name__ == '__main__':
    unittest.main()