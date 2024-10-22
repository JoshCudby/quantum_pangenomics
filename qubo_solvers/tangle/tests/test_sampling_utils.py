import unittest
from utils.sampling_utils import get_node_visits

class TestSamplingUtilsMethods(unittest.TestCase):
    
    def setUp(self):
        self.num_nodes = 4
        self.t_max = 6
        
    def test_get_node_visits(self):
        bqm_variables = [(x, t) for x in range(self.num_nodes) for t in range(self.t_max)]
        sample_like = {var : 0 for var in bqm_variables}
        opt_path = [(0, 0), (1, 1), (2, 2), (1, 3), (3, 4), (4, 5)]
        for node in opt_path:
            sample_like[node] = 1
            
        self.assertEqual(get_node_visits(sample_like), {0: 1, 1: 2, 2: 1, 3: 1, 4: 1})
        
        
if __name__ == '__main__':
    unittest.main()