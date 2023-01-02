import unittest
from MOO.MOFEA import MOFEA
from MOO.MOEA import NSGA2
from FEA.factorarchitecture import *
from optimizationproblems.knapsack import *
from utilities.util import add_method


class TestMOFEA(unittest.TestCase):
    def setUp(self) -> None:
        dim = 20
        n_obj = 3
        ks = Knapsack(number_of_items=dim, max_nr_items=dim, nr_objectives=n_obj, nr_constraints=1)
        algorithm = NSGA2

        @add_method(NSGA2)
        def calc_fitness(variables, gs=None, factor=None):
            if gs is not None and factor is not None:
                full_solution = [x for x in gs.variables]
                for i, x in zip(factor, variables):
                    full_solution[i] = x
            else:
                full_solution = variables
            ks.set_fitness_multi_knapsack(full_solution)
            return ks.objective_values

        self.feamoo = MOFEA(10, 10, 10, base_alg=algorithm, dimensions=dim,
                            combinatorial_options=[0, 1], ref_point=ks.ref_point)
        self.factors = [[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 8], [8, 9, 0]]
        self.feamoo.subpopulations = self.feamoo.initialize_moo_subpopulations(factors=self.factors)

    def test_compete(self):
        self.feamoo.compete()

    def test_share(self):
        self.feamoo.share_solution()

    # def test_init_population(self):
    #     subpopulations = self.feamoo.initialize_moo_subpopulations(factors=self.factors)
    #     for i, pop in enumerate(subpopulations):
    #         self.assertIsInstance(pop, NSGA2)
    #         self.assertEqual(len(pop.curr_population), len(self.factors[i]))
    #     self.assertEqual(len(subpopulations), len(self.factors))


if __name__ == '__main__':
    unittest.main()