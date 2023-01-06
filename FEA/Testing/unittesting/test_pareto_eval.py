import unittest

import numpy as np
from pymoo.util.dominator import Dominator

from utilities.util import compare_solutions, PopulationMember


class TestEvaluationMetrics(unittest.TestCase):
    def setUp(self) -> None:
        dim = 20
        n_obj = 3

    def test_spread_indicator(self):
        pass
        #self.assertAlmostEqual()


    def test_pymoo_dominance(self):
        fitnesses = np.array([np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([0.0, 1.0])])
        domination_matrix = Dominator().calc_domination_matrix(fitnesses)
        print(domination_matrix)

    def test_dominance(self):
        s1 = PopulationMember(variables=[0, 0, 0, 0], fitness=[0.0, 0.0])
        s2 = PopulationMember(variables=[0, 0, 0, 0], fitness=[1.0, 1.0])
        s3 = PopulationMember(variables=[0, 0, 0, 0], fitness=[0.0, 1.0])

        self.assertEqual(-1, compare_solutions(s1, s2))
        self.assertEqual(1, compare_solutions(s2, s1))
        self.assertEqual(-1, compare_solutions(s1, s3))
        self.assertEqual(1, compare_solutions(s3, s1))

    def test_nondominance(self):
        s1 = PopulationMember(variables=[0, 0, 0, 0], fitness=[0.0, 1.0])
        s2 = PopulationMember(variables=[0, 0, 0, 0], fitness=[0.5, 0.5])
        s3 = PopulationMember(variables=[0, 0, 0, 0], fitness=[1.0, 0.0])

        self.assertEqual(0, compare_solutions(s1, s2))
        self.assertEqual(0, compare_solutions(s2, s1))
        self.assertEqual(0, compare_solutions(s2, s3))
        self.assertEqual(0, compare_solutions(s3, s2))
        self.assertEqual(0, compare_solutions(s1, s3))
        self.assertEqual(0, compare_solutions(s3, s1))

if __name__ == '__main__':
    unittest.main()