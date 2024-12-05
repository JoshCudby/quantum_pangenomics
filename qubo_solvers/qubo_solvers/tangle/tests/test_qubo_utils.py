import unittest
import networkx as nx
from utils.qubo_utils import _tangle_problem_bqm


def toy_graph(exact_solution=True):
    weight_1 = 3 if exact_solution else 4
        
    g = nx.DiGraph()
    g.add_nodes_from([
        (0, {"weight": 1}),
        (1, {"weight": weight_1}),
        (2, {"weight": 1}),
        (3, {"weight": 1}),
        (4, {"weight": 1}),
    ])
    g.add_edges_from([
        (0, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 3), (2, 4), (3, 1), (3, 2), (3, 4),
    ])
    return g


class TestQuboUtilsMethods(unittest.TestCase):
    
    def setUp(self):
        self.graph = toy_graph(True)
        self.num_nodes = len(list(self.graph.nodes))
        self.t_max = sum(self.graph.nodes.data()[i]["weight"] for i in range(len(self.graph.nodes))) + 1
        
        self.P = 10
        self.mu = 0.5
        self.lamda = [0] * len(self.graph.nodes)
        
        self.bqm = _tangle_problem_bqm(self.graph, self.lamda, self.mu, self.P)

    def test_tangle_problem_bqm_vars(self):
        expected_variables = [(x, t) for x in range(self.num_nodes + 1) for t in range(self.t_max)]
        self.assertEqual(
            sorted(self.bqm.variables, key=lambda e: [e[0], e[1]]), 
            sorted(expected_variables, key=lambda e: [e[0], e[1]])
        )
        
    def test_tangle_problem_bqm_good_interaction(self):
        self.assertEqual(
            self.bqm.get_quadratic((0, 0), (1, 1)),
            -1
        )
        self.assertEqual(
            self.bqm.get_quadratic((3, 2), (4, 3)),
            -1
        )
    
    def test_tangle_problem_bqm_bad_interaction(self):
        self.assertEqual(
            self.bqm.get_quadratic((0, 0), (4, 1)),
            self.P
        )
    
    def test_tangle_problem_bqm_neutral_interaction(self):
        self.assertEqual(
            self.bqm.get_quadratic((4, 0), (5, 1)),
            0
        )
        self.assertEqual(
            self.bqm.get_quadratic((5, 0), (5, 1)),
            0
        )
        
    def test_tangle_problem_bqm_multiple_locations_interaction(self):
        self.assertEqual(
            self.bqm.get_quadratic((2, 1), (3, 1)),
            2 * self.P
        )
        self.assertEqual(
            self.bqm.get_quadratic((0, 2), (5, 2)),
            2 * self.P
        )
        
    def test_tangle_problem_bqm_node_weight_interaction(self):
        self.assertEqual(
            self.bqm.get_quadratic((2, 1), (2, 2)),
            self.mu + self.P
        )
        self.assertEqual(
            self.bqm.get_quadratic((0, 2), (0, 3)),
            self.mu + self.P
        )
        
    def test_tangle_problem_bqm_linear_start(self):
        self.assertEqual(
            self.bqm.get_linear((5, self.t_max - 1)),
            -2 * self.P
        )
        
    def test_tangle_problem_bqm_linear_end(self):
        self.assertEqual(
            self.bqm.get_linear((5, self.t_max - 1)),
            -2 * self.P
        )
    
    def test_tangle_problem_bqm_linear_middle(self):
        self.assertEqual(
            self.bqm.get_linear((2, 2)),
            -1 * self.P + self.mu / 2 * (1 - 2 * self.graph.nodes.data()[0]["weight"]) + self.lamda[0]
        )
        
        
if __name__ == '__main__':
    unittest.main()