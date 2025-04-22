import unittest
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import Operator

import numpy as np
import networkx as nx

from scipy.linalg import expm

from functools import reduce

from qiskit_aer import AerSimulator
from qiskit_prog_qaoa.utils.circuit_utils import (
    is_equal_to, controlled_copy_with_swap, 
    compute_next_nodes, penalise_graph_steps, penalise_graph_end_steps, uncompute_next_nodes,    
    compute_count, penalise_count, uncompute_count, 
    get_constraint_circuit, get_objective_circuit,
    uniform_over_range, state_prep,
    get_mixer_operator
)


def toy_graph():
    g = nx.Graph()
    g.add_nodes_from([
        (1, {"weight": 1}),
        (2, {"weight": 1}),
        (3, {"weight": 1}),
    ])
    g.add_edges_from([
        (1,2), (2,3)
    ])
    return g


def get_single_bitstring_mapping(
    input_str: str,
    circuit_to_run: QuantumCircuit,
    simulator: AerSimulator
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
    t_circ = transpile(circ, simulator, optimization_level=3)
    result = simulator.run(t_circ).result()
    start_ket = np.binary_repr(np.nonzero(result.data()['start'].data)[0][0], len(input_str))[::-1]
    end_ket = np.binary_repr(np.nonzero(result.data()['end'].data)[0][0], len(input_str))[::-1]
    return start_ket, end_ket


def get_single_bitstring_mapping_with_amplitudes(
    input_str: str,
    circuit_to_run: QuantumCircuit,
    simulator: AerSimulator
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
    t_circ = transpile(circ, simulator, optimization_level=3)
    result = simulator.run(t_circ).result()
    start_data = result.data()['start'].data
    start_ket = np.binary_repr(np.nonzero(start_data)[0][0], len(input_str))[::-1]
    start_amplitude = start_data[np.nonzero(start_data)[0][0]]

    end_data = result.data()['end'].data
    end_ket = np.binary_repr(np.nonzero(end_data)[0][0], len(input_str))[::-1]
    end_amplitude = end_data[np.nonzero(end_data)[0][0]]

    return start_ket, start_amplitude, end_ket, end_amplitude


def get_final_statevector(    
    circuit_to_run: QuantumCircuit,
    simulator: AerSimulator
):
    circuit_to_run.save_statevector('end')
    t_circ = transpile(circuit_to_run, simulator, optimization_level=3)
    result = simulator.run(t_circ).result()
    return result.data()['end'].data
    


class TestIsEqualTo(unittest.TestCase):
    def test_is_equal_to_2_1(self):        
        equal = is_equal_to(2, 1)
        unitary = Operator.from_circuit(equal).data
        
        want = np.array([
            [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j]
        ])
        self.assertTrue(
            np.allclose(unitary, want)
        )

    def test_is_equal_to_2_2(self):        
        equal = is_equal_to(2, 2)
        unitary = Operator.from_circuit(equal).data
        
        want = np.array([
            [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j]
        ])
        self.assertTrue(
            np.allclose(unitary, want)
        )
        
class TestCcSwap(unittest.TestCase):
    def setUp(self):
        self.n = 3
        self.K = 3
        self.num_qubits = 1 + (self.K+1) * self.n
        self.cc = controlled_copy_with_swap(self.n, self.K)
        self.simulator = AerSimulator()

    def test_cc_swap_identity_if_control_off(self):
        start_ket, end_ket = get_single_bitstring_mapping(
            '0000001101000',
            self.cc,
            self.simulator
        )
        self.assertEqual(
            start_ket,
            '0000001101000'
        )
        self.assertEqual(
            end_ket,
            '0000001101000'
        )

        start_ket, end_ket = get_single_bitstring_mapping(
            '0111001101000',
            self.cc,
            self.simulator
        )
        self.assertEqual(
            start_ket,
            '0111001101000'
        )
        self.assertEqual(
            end_ket,
            '0111001101000'
        )

        start_ket, end_ket = get_single_bitstring_mapping(
            '0111111101000',
            self.cc,
            self.simulator
        )
        self.assertEqual(
            start_ket,
            '0111111101000'
        )
        self.assertEqual(
            end_ket,
            '0111111101000'
        )

    def test_cc_swap_shuffles_copy_registers_if_control_on(self):
        start_ket, end_ket = get_single_bitstring_mapping(
            '1000001101000',
            self.cc,
            self.simulator
        )
        self.assertEqual(
            start_ket,
            '1000001101000'
        )
        self.assertEqual(
            end_ket,
            '1000000001101'
        )

    def test_cc_swap_copies_register_if_control_on(self):
        start_ket, end_ket = get_single_bitstring_mapping(
            '1011000000000',
            self.cc,
            self.simulator
        )
        self.assertEqual(
            start_ket,
            '1011000000000'
        )
        self.assertEqual(
            end_ket,
            '1011011000000'
        )

        start_ket, end_ket = get_single_bitstring_mapping(
            '1101000000000',
            self.cc,
            self.simulator
        )
        self.assertEqual(
            start_ket,
            '1101000000000'
        )
        self.assertEqual(
            end_ket,
            '1101101000000'
        )


class TestComputeNextNodes(unittest.TestCase):
    def setUp(self):
        self.simulator = AerSimulator()
        self.n = 3
        self.T = 3
        self.K = 3
        ceil_log_n2 = int(np.ceil(np.log2(self.n+2)))
        circuit = QuantumCircuit()

        registers = {f'solution_{t}' : QuantumRegister(ceil_log_n2, name=f'solution_{t}') for t in range(self.T)}
        registers.update({f'next_node_{k}': QuantumRegister(ceil_log_n2, name=f'next_node_{k}') for k in range(self.K)})
        registers.update({'flag': QuantumRegister(1, name='flag')})


        for register in registers.values():
            circuit.add_register(register)
        self.nn = compute_next_nodes(circuit, registers, 1, self.n, self.K, self.T)

    
    def test_compute_nn_copies_one_next_node(self):
        start_ket, end_ket = get_single_bitstring_mapping(
            '0010111000000000000',
            self.nn,
            self.simulator
        )
        self.assertEqual(
            start_ket,
            '0010111000000000000'
        )
        self.assertEqual(
            end_ket,
            '0010111000110000000'
        )


    def test_compute_nn_copies_several_next_nodes(self):
        start_ket, end_ket = get_single_bitstring_mapping(
            '0010011110000000000',
            self.nn,
            self.simulator
        )
        self.assertEqual(
            start_ket,
            '0010011110000000000'
        )
        self.assertEqual(
            end_ket,
            '0010011111110010000'
        )


class TestUncomputeNextNodes(unittest.TestCase):
    def setUp(self):
        self.simulator = AerSimulator()
        self.n = 3
        self.T = 3
        self.K = 3
        ceil_log_n2 = int(np.ceil(np.log2(self.n+2)))
        circuit = QuantumCircuit()

        registers = {f'solution_{t}' : QuantumRegister(ceil_log_n2, name=f'solution_{t}') for t in range(self.T)}
        registers.update({f'next_node_{k}': QuantumRegister(ceil_log_n2, name=f'next_node_{k}') for k in range(self.K)})
        registers.update({'flag': QuantumRegister(1, name='flag')})


        for register in registers.values():
            circuit.add_register(register)
        self.nn = uncompute_next_nodes(circuit, registers, 1, self.n, self.K, self.T)

    
    def test_uncompute_nn_uncopies_one_next_node(self):
        start_ket, end_ket = get_single_bitstring_mapping(
            '0010111000110000000',
            self.nn,
            self.simulator
        )
        self.assertEqual(
            start_ket,
            '0010111000110000000'
        )
        self.assertEqual(
            end_ket,
            '0010111000000000000'
        )


    def test_uncompute_nn_uncopies_several_next_nodes(self):
        start_ket, end_ket = get_single_bitstring_mapping(
            '0010011111110010000',
            self.nn,
            self.simulator
        )
        self.assertEqual(
            start_ket,
            '0010011111110010000'
        )
        self.assertEqual(
            end_ket,
            '0010011110000000000'
        )


class TestPenaliseGraphSteps(unittest.TestCase):
    def setUp(self):
        self.simulator = AerSimulator()
        self.n = 3
        self.T = 3
        self.K = 3
        ceil_log_n2 = int(np.ceil(np.log2(self.n+2)))
        self.parameter = np.pi/16
        self.graph = toy_graph()
        circuit = QuantumCircuit()

        registers = {f'solution_{t}' : QuantumRegister(ceil_log_n2, name=f'solution_{t}') for t in range(self.T)}
        registers.update({f'next_node_{k}': QuantumRegister(ceil_log_n2, name=f'next_node_{k}') for k in range(self.K)})
        registers.update({'flag': QuantumRegister(1, name='flag')})


        for register in registers.values():
            circuit.add_register(register)
        self.gs = penalise_graph_steps(circuit, registers, 1, self.parameter, self.graph, self.n, self.K)

    def test_no_penalty_no_next_nodes(self):
        start_ket, start_amplitude, end_ket, end_amplitude = get_single_bitstring_mapping_with_amplitudes(
            '0110111000000000000',
            self.gs,
            self.simulator
        )
        self.assertEqual(
            start_ket,
            '0110111000000000000'
        )
        self.assertAlmostEqual(
            start_amplitude,
            1. + 0.j
        )
        self.assertEqual(
            end_ket,
            '0110111000000000000'
        )
        self.assertAlmostEqual(
            end_amplitude,
            1. + 0.j
        )

    
    def test_no_penalty_valid_graph_steps(self):
        start_ket, start_amplitude, end_ket, end_amplitude = get_single_bitstring_mapping_with_amplitudes(
            '0010111000100000000',
            self.gs,
            self.simulator
        )
        self.assertEqual(
            start_ket,
            '0010111000100000000'
        )
        self.assertAlmostEqual(
            start_amplitude,
            1. + 0.j
        )
        self.assertEqual(
            end_ket,
            '0010111000100000000'
        )
        self.assertAlmostEqual(
            end_amplitude,
            1. + 0.j
        )

    def test_penalty_broken_graph_step(self):
        start_ket, start_amplitude, end_ket, end_amplitude = get_single_bitstring_mapping_with_amplitudes(
            '0010110110110000000',
            self.gs,
            self.simulator
        )
        self.assertEqual(
            start_ket,
            '0010110110110000000'
        )
        self.assertAlmostEqual(
            start_amplitude,
            1. + 0.j
        )
        self.assertEqual(
            end_ket,
            '0010110110110000000'
        )
        self.assertAlmostEqual(
            end_amplitude,
            np.e ** (1j * self.parameter)
        )


    def test_penalty_broken_graph_steps(self):
        start_ket, start_amplitude, end_ket, end_amplitude = get_single_bitstring_mapping_with_amplitudes(
            '0010010110110010000',
            self.gs,
            self.simulator
        )
        self.assertEqual(
            start_ket,
            '0010010110110010000'
        )
        self.assertAlmostEqual(
            start_amplitude,
            1. + 0.j
        )
        self.assertEqual(
            end_ket,
            '0010010110110010000'
        )
        self.assertAlmostEqual(
            end_amplitude,
            np.e ** (2j * self.parameter)
        )


class TestPenaliseGraphEndSteps(unittest.TestCase):
    def setUp(self):
        self.simulator = AerSimulator()
        self.n = 3
        self.T = 3
        self.K = 3
        ceil_log_n2 = int(np.ceil(np.log2(self.n+2)))
        self.parameter = np.pi/16
        circuit = QuantumCircuit()

        registers = {f'solution_{t}' : QuantumRegister(ceil_log_n2, name=f'solution_{t}') for t in range(self.T)}
        registers.update({f'next_node_{k}': QuantumRegister(ceil_log_n2, name=f'next_node_{k}') for k in range(self.K)})
        registers.update({'flag': QuantumRegister(1, name='flag')})


        for register in registers.values():
            circuit.add_register(register)
        self.gs = penalise_graph_end_steps(circuit, registers, self.parameter, self.n, self.K)

    def test_no_penalty_no_next_nodes(self):
        start_ket, start_amplitude, end_ket, end_amplitude = get_single_bitstring_mapping_with_amplitudes(
            '0110111000000000000',
            self.gs,
            self.simulator
        )
        self.assertEqual(
            start_ket,
            '0110111000000000000'
        )
        self.assertAlmostEqual(
            start_amplitude,
            1. + 0.j
        )
        self.assertEqual(
            end_ket,
            '0110111000000000000'
        )
        self.assertAlmostEqual(
            end_amplitude,
            1. + 0.j
        )

    
    def test_no_penalty_valid_graph_steps(self):
        start_ket, start_amplitude, end_ket, end_amplitude = get_single_bitstring_mapping_with_amplitudes(
            '1001001001001000000',
            self.gs,
            self.simulator
        )
        self.assertEqual(
            start_ket,
            '1001001001001000000'
        )
        self.assertAlmostEqual(
            start_amplitude,
            1. + 0.j
        )
        self.assertEqual(
            end_ket,
            '1001001001001000000'
        )
        self.assertAlmostEqual(
            end_amplitude,
            1. + 0.j
        )

    def test_penalty_broken_graph_step(self):
        start_ket, start_amplitude, end_ket, end_amplitude = get_single_bitstring_mapping_with_amplitudes(
            '1001000010011000000',
            self.gs,
            self.simulator
        )
        self.assertEqual(
            start_ket,
            '1001000010011000000'
        )
        self.assertAlmostEqual(
            start_amplitude,
            1. + 0.j
        )
        self.assertEqual(
            end_ket,
            '1001000010011000000'
        )
        self.assertAlmostEqual(
            end_amplitude,
            np.e ** (1j * self.parameter)
        )


    def test_penalty_broken_graph_steps(self):
        start_ket, start_amplitude, end_ket, end_amplitude = get_single_bitstring_mapping_with_amplitudes(
            '0000000001000100010',
            self.gs,
            self.simulator
        )
        self.assertEqual(
            start_ket,
            '0000000001000100010'
        )
        self.assertAlmostEqual(
            start_amplitude,
            1. + 0.j
        )
        self.assertEqual(
            end_ket,
            '0000000001000100010'
        )
        self.assertAlmostEqual(
            end_amplitude,
            np.e ** (2j * self.parameter)
        )


class TestComputeCount(unittest.TestCase):
    def setUp(self):
        self.simulator = AerSimulator()
        self.n = 3
        self.T = 3
        self.K = 3
        ceil_log_n2 = int(np.ceil(np.log2(self.n+2)))
        circuit = QuantumCircuit()
        ceil_log_K1 = int(np.ceil(np.log2(self.K+1)))

        registers = {f'solution_{t}' : QuantumRegister(ceil_log_n2, name=f'solution_{t}') for t in range(self.T)}
        registers.update({'flag': QuantumRegister(1, name='flag')})
        registers.update({'count': QuantumRegister(ceil_log_K1, name='count')})


        for register in registers.values():
            circuit.add_register(register)
        self.cc = compute_count(circuit, registers, 1, self.n, self.K, self.T)

    def test_compute_count_counts_no_visits(self):
        start_ket, end_ket = get_single_bitstring_mapping(
            '010010010000',
            self.cc,
            self.simulator
        )
        self.assertEqual(
            start_ket,
            '010010010000'
        )
        self.assertEqual(
            end_ket,
            '010010010000'
        )

    
    def test_compute_count_counts_one_visit(self):
        start_ket, end_ket = get_single_bitstring_mapping(
            '001010010000',
            self.cc,
            self.simulator
        )
        self.assertEqual(
            start_ket,
            '001010010000'
        )
        self.assertEqual(
            end_ket,
            '001010010001'
        )

    def test_compute_count_counts_two_visits(self):
        start_ket, end_ket = get_single_bitstring_mapping(
            '001001010000',
            self.cc,
            self.simulator
        )
        self.assertEqual(
            start_ket,
            '001001010000'
        )
        self.assertEqual(
            end_ket,
            '001001010010'
        )

    def test_compute_count_counts_three_visits(self):
        start_ket, end_ket = get_single_bitstring_mapping(
            '001001001000',
            self.cc,
            self.simulator
        )
        self.assertEqual(
            start_ket,
            '001001001000'
        )
        self.assertEqual(
            end_ket,
            '001001001011'
        )


class TestUncomputeCount(unittest.TestCase):
    def setUp(self):
        self.simulator = AerSimulator()
        self.n = 3
        self.T = 3
        self.K = 3
        ceil_log_n2 = int(np.ceil(np.log2(self.n+2)))
        circuit = QuantumCircuit()
        ceil_log_K1 = int(np.ceil(np.log2(self.K+1)))

        registers = {f'solution_{t}' : QuantumRegister(ceil_log_n2, name=f'solution_{t}') for t in range(self.T)}
        registers.update({'flag': QuantumRegister(1, name='flag')})
        registers.update({'count': QuantumRegister(ceil_log_K1, name='count')})


        for register in registers.values():
            circuit.add_register(register)
        self.cc = uncompute_count(circuit, registers, 1, self.n, self.K, self.T)

    def test_uncompute_count_uncounts_no_visits(self):
        start_ket, end_ket = get_single_bitstring_mapping(
            '010010010000',
            self.cc,
            self.simulator
        )
        self.assertEqual(
            start_ket,
            '010010010000'
        )
        self.assertEqual(
            end_ket,
            '010010010000'
        )

    
    def test_uncompute_count_uncounts_one_visit(self):
        start_ket, end_ket = get_single_bitstring_mapping(
            '001010010001',
            self.cc,
            self.simulator
        )
        self.assertEqual(
            start_ket,
            '001010010001'
        )
        self.assertEqual(
            end_ket,
            '001010010000'
        )

    def test_uncompute_count_uncounts_two_visits(self):
        start_ket, end_ket = get_single_bitstring_mapping(
            '001001010010',
            self.cc,
            self.simulator
        )
        self.assertEqual(
            start_ket,
            '001001010010'
        )
        self.assertEqual(
            end_ket,
            '001001010000'
        )

    def test_uncompute_count_uncounts_three_visits(self):
        start_ket, end_ket = get_single_bitstring_mapping(
            '001001001011',
            self.cc,
            self.simulator
        )
        self.assertEqual(
            start_ket,
            '001001001011'
        )
        self.assertEqual(
            end_ket,
            '001001001000'
        )


class TestPenaliseCount(unittest.TestCase):
    def setUp(self):
        self.simulator = AerSimulator()
        self.n = 3
        self.T = 3
        self.K = 3
        ceil_log_K1 = int(np.ceil(np.log2(self.K+1)))
        ceil_log_n2 = int(np.ceil(np.log2(self.n+2)))
        self.parameter = np.pi/16
        self.graph = toy_graph()
        circuit = QuantumCircuit()

        registers = {f'solution_{t}' : QuantumRegister(ceil_log_n2, name=f'solution_{t}') for t in range(self.T)}
        registers.update({'flag': QuantumRegister(1, name='flag')})
        registers.update({'count': QuantumRegister(ceil_log_K1, name='count')})


        for register in registers.values():
            circuit.add_register(register)
        self.gs = penalise_count(circuit, registers, 1, self.parameter, self.graph, self.K)


    def test_no_penalty_count_is_weight(self):
        start_ket, start_amplitude, end_ket, end_amplitude = get_single_bitstring_mapping_with_amplitudes(
            '001010010001',
            self.gs,
            self.simulator
        )
        self.assertEqual(
            start_ket,
            '001010010001'
        )
        self.assertAlmostEqual(
            start_amplitude,
            1. + 0.j
        )
        self.assertEqual(
            end_ket,
            '001010010001'
        )
        self.assertAlmostEqual(
            end_amplitude,
            1. + 0.j
        )

    def test_penalty_count_1_under_weight(self):
        start_ket, start_amplitude, end_ket, end_amplitude = get_single_bitstring_mapping_with_amplitudes(
            '010010010000',
            self.gs,
            self.simulator
        )
        self.assertEqual(
            start_ket,
            '010010010000'
        )
        self.assertAlmostEqual(
            start_amplitude,
            1. + 0.j
        )
        self.assertEqual(
            end_ket,
            '010010010000'
        )
        self.assertAlmostEqual(
            end_amplitude,
            np.e ** (1j * self.parameter)
        )

    
    def test_penalty_count_1_over_weight(self):
        start_ket, start_amplitude, end_ket, end_amplitude = get_single_bitstring_mapping_with_amplitudes(
            '001001010010',
            self.gs,
            self.simulator
        )
        self.assertEqual(
            start_ket,
            '001001010010'
        )
        self.assertAlmostEqual(
            start_amplitude,
            1. + 0.j
        )
        self.assertEqual(
            end_ket,
            '001001010010'
        )
        self.assertAlmostEqual(
            end_amplitude,
            np.e ** (1j * self.parameter)
        )

    def test_penalty_count_2_over_weight(self):
        start_ket, start_amplitude, end_ket, end_amplitude = get_single_bitstring_mapping_with_amplitudes(
            '001001001011',
            self.gs,
            self.simulator
        )
        self.assertEqual(
            start_ket,
            '001001001011'
        )
        self.assertAlmostEqual(
            start_amplitude,
            1. + 0.j
        )
        self.assertEqual(
            end_ket,
            '001001001011'
        )
        self.assertAlmostEqual(
            end_amplitude,
            np.e ** (4j * self.parameter)
        )


class TestConstraintCircuit(unittest.TestCase):
    def setUp(self):
        self.simulator = AerSimulator()
        self.n = 3
        self.T = 3
        self.K = 3
        self.parameter = np.pi/16
        self.graph = toy_graph()

        self.cc = get_constraint_circuit(self.n, self.K, self.T, self.graph, self.parameter, None)


    def test_constraint_circuit_no_penalty(self):
        start_ket, start_amplitude, end_ket, end_amplitude = get_single_bitstring_mapping_with_amplitudes(
            '0010101000000000000',
            self.cc,
            self.simulator
        )
        self.assertEqual(
            start_ket,
            '0010101000000000000'
        )
        self.assertAlmostEqual(
            start_amplitude,
            1. + 0.j
        )
        self.assertEqual(
            end_ket,
            '0010101000000000000'
        )
        self.assertAlmostEqual(
            end_amplitude,
            1. + 0.j
        )

    def test_constraint_circuit_one_broken_graph_step(self):
        start_ket, start_amplitude, end_ket, end_amplitude = get_single_bitstring_mapping_with_amplitudes(
            '0100101000000000000',
            self.cc,
            self.simulator
        )
        self.assertEqual(
            start_ket,
            '0100101000000000000'
        )
        self.assertAlmostEqual(
            start_amplitude,
            1. + 0.j
        )
        self.assertEqual(
            end_ket,
            '0100101000000000000'
        )
        self.assertAlmostEqual(
            end_amplitude,
            np.e ** (1j * self.parameter)
        )

    
    def test_constraint_circuit_two_broken_graph_steps(self):
        start_ket, start_amplitude, end_ket, end_amplitude = get_single_bitstring_mapping_with_amplitudes(
            '0100100100000000000',
            self.cc,
            self.simulator
        )
        self.assertEqual(
            start_ket,
            '0100100100000000000'
        )
        self.assertAlmostEqual(
            start_amplitude,
            1. + 0.j
        )
        self.assertEqual(
            end_ket,
            '0100100100000000000'
        )
        self.assertAlmostEqual(
            end_amplitude,
            np.e ** (2j * self.parameter)
        )

    def test_constraint_circuit_broken_graph_end_step(self):
        start_ket, start_amplitude, end_ket, end_amplitude = get_single_bitstring_mapping_with_amplitudes(
            '0101000100000000000',
            self.cc,
            self.simulator
        )
        self.assertEqual(
            start_ket,
            '0101000100000000000'
        )
        self.assertAlmostEqual(
            start_amplitude,
            1. + 0.j
        )
        self.assertEqual(
            end_ket,
            '0101000100000000000'
        )
        self.assertAlmostEqual(
            end_amplitude,
            np.e ** (1j * self.parameter)
        )

    def test_constraint_circuit_broken_graph_and_graph_end_step(self):
        start_ket, start_amplitude, end_ket, end_amplitude = get_single_bitstring_mapping_with_amplitudes(
            '1000010110000000000',
            self.cc,
            self.simulator
        )
        self.assertEqual(
            start_ket,
            '1000010110000000000'
        )
        self.assertAlmostEqual(
            start_amplitude,
            1. + 0.j
        )
        self.assertEqual(
            end_ket,
            '1000010110000000000'
        )
        self.assertAlmostEqual(
            end_amplitude,
            np.e ** (2j * self.parameter)
        )


class TestObjectiveCircuit(unittest.TestCase):
    def setUp(self):
        self.simulator = AerSimulator()
        self.n = 3
        self.T = 3
        self.K = 3
        self.parameter = np.pi/16
        self.graph = toy_graph()

        self.oc = get_objective_circuit(self.n, self.K, self.T, self.graph, self.parameter, None)


    def test_objective_circuit_no_penalty(self):
        start_ket, start_amplitude, end_ket, end_amplitude = get_single_bitstring_mapping_with_amplitudes(
            '001010011000',
            self.oc,
            self.simulator
        )
        self.assertEqual(
            start_ket,
            '001010011000'
        )
        self.assertAlmostEqual(
            start_amplitude,
            1. + 0.j
        )
        self.assertEqual(
            end_ket,
            '001010011000'
        )
        self.assertAlmostEqual(
            end_amplitude,
            1. + 0.j
        )

    def test_objective_circuit_2_nodes_off_by_1(self):
        start_ket, start_amplitude, end_ket, end_amplitude = get_single_bitstring_mapping_with_amplitudes(
            '010010011000',
            self.oc,
            self.simulator
        )
        self.assertEqual(
            start_ket,
            '010010011000'
        )
        self.assertAlmostEqual(
            start_amplitude,
            1. + 0.j
        )
        self.assertEqual(
            end_ket,
            '010010011000'
        )
        self.assertAlmostEqual(
            end_amplitude,
            np.e ** (2j * self.parameter)
        )

    
    def test_objective_circuit_1_nodes_off_by_2_2_nodes_off_by_1(self):
        start_ket, start_amplitude, end_ket, end_amplitude = get_single_bitstring_mapping_with_amplitudes(
            '010010010000',
            self.oc,
            self.simulator
        )
        self.assertEqual(
            start_ket,
            '010010010000'
        )
        self.assertAlmostEqual(
            start_amplitude,
            1. + 0.j
        )
        self.assertEqual(
            end_ket,
            '010010010000'
        )
        self.assertAlmostEqual(
            end_amplitude,
            np.e ** (6j * self.parameter)
        )


class TestUniformOverRange(unittest.TestCase):
    def setUp(self):
        self.n = 3
        self.simulator = AerSimulator()
        
    def test_uniform_M_2(self):
        circ = uniform_over_range(self.n, 2)
        final_sv = get_final_statevector(circ, self.simulator)

        want = np.zeros(2 ** self.n)
        want[1:3] = 2 ** -0.5

        self.assertTrue(
            np.allclose(
                final_sv, want
            )
        )

    def test_uniform_M_3(self):
        circ = uniform_over_range(self.n, 3)
        final_sv = get_final_statevector(circ, self.simulator)

        want = np.zeros(2 ** self.n)
        want[1:4] = 3 ** -0.5

        self.assertTrue(
            np.allclose(
                final_sv, want
            )
        )

    def test_uniform_M_6(self):
        circ = uniform_over_range(self.n, 6)
        final_sv = get_final_statevector(circ, self.simulator)

        want = np.zeros(2 ** self.n)
        want[1:7] = 6 ** -0.5

        self.assertTrue(
            np.allclose(
                final_sv, want
            )
        )

    def test_uniform_M_7(self):
        circ = uniform_over_range(self.n, 7)
        final_sv = get_final_statevector(circ, self.simulator)

        want = np.zeros(2 ** self.n)
        want[1:8] = 7 ** -0.5

        self.assertTrue(
            np.allclose(
                final_sv, want
            )
        )


class TestStatePrep(unittest.TestCase):
    def setUp(self):
        self.n = 3
        self.T = 2
        self.simulator = AerSimulator()
        
    def test_state_prep(self):
        circ = state_prep(self.n, self.T)
        final_sv = get_final_statevector(circ, self.simulator)

        want = np.zeros(2 ** self.n)
        want[1:self.n+2] = (self.n+1) ** -0.5
        tensor_want = np.kron(want, want)
        self.assertTrue(
            np.allclose(
                final_sv, tensor_want
            )
        )

    
class TestMixerOperator(unittest.TestCase):
    def setUp(self):
        self.n = 2
        self.T = 2
        self.parameter = np.pi/16

    def test_mixer_operator(self):
        mixer = get_mixer_operator(self.n, self.T, self.parameter)
        mixer_data = Operator.from_circuit(mixer).data

        unif = np.zeros(2 ** int(np.ceil(np.log(self.n+2))))
        unif[1:self.n+2] = (self.n+1) ** -0.5

        feasible = reduce(
            np.kron,
            [unif] * self.T
        )
        feasible_ketbra = np.outer(feasible, feasible)
        want = expm(-1j * self.parameter * feasible_ketbra)

        self.assertTrue(
            np.allclose(mixer_data, want)
        )



if __name__ == '__main__':
    unittest.main()