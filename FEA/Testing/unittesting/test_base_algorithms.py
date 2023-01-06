import random
import unittest
from basealgorithms.ga import *
from basealgorithms.pso import *
from utilities.util import add_method


class TestGAContinuous(unittest.TestCase):
    def setUp(self) -> None:
        population_size = 10
        dimensions = 10
        self.ga = GA(dimensions, population_size, continuous_var_space=True, upper_value_limit=50)

        @add_method(GA)
        def calc_fitness(variables):
            return sum(variables)

        self.ga.initialize_population()
        random_member_to_check = random.randint(1, self.ga.population_size - 1)

    def test_population_initialization(self):
        self.assertEqual(len(self.ga.population), self.ga.population_size)

        random_member_to_check = random.randint(1, self.ga.population_size - 1)
        self.assertEqual(len(self.ga.population[random_member_to_check].variables), self.ga.dimensions)

    def test_tournament_selection(self):
        chosen_solution, idx = self.ga.tournament_selection(self.ga.population)
        self.assertIn(chosen_solution, self.ga.population)
        self.assertEqual(chosen_solution, self.ga.population[idx])

    def test_single_crossover(self):
        sol1 = [random.randrange(0, self.ga.upper_value_limit) for x in range(self.ga.dimensions)]
        sol2 = [random.randrange(0, self.ga.upper_value_limit) for x in range(self.ga.dimensions)]
        third_solution = [(x + y) / 2 for (x, y) in zip(sol1, sol2)]
        self.ga.crossover_type = 'single'
        crossedover = self.ga.crossover(sol1, sol2, crossover_rate=1)
        self.assertEqual(crossedover[0][0], third_solution[0])
        self.assertEqual(crossedover[1][-1], third_solution[-1])

    def test_multi_crossover(self):
        sol1 = [random.randrange(0, self.ga.upper_value_limit) for x in range(self.ga.dimensions)]
        sol2 = [random.randrange(0, self.ga.upper_value_limit) for x in range(self.ga.dimensions)]
        third_solution = [(x + y) / 2 for (x, y) in zip(sol1, sol2)]
        self.ga.crossover_type = 'multi'
        crossedover = self.ga.crossover(sol1, sol2, crossover_rate=1)
        self.assertEqual(crossedover[1][0], third_solution[0])
        self.assertEqual(crossedover[1][-1], third_solution[-1])

    def test_uniform_crossover(self):
        sol1 = [random.randrange(0, self.ga.upper_value_limit) for x in range(self.ga.dimensions)]
        sol2 = [random.randrange(0, self.ga.upper_value_limit) for x in range(self.ga.dimensions)]
        third_solution = [(x + y) / 2 for (x, y) in zip(sol1, sol2)]
        self.ga.crossover_type = 'uniform'
        crossedover = self.ga.crossover(sol1, sol2, crossover_rate=1)
        self.assertEqual(crossedover[0], third_solution)

    def test_continuous_mutation(self):
        sol = [random.randrange(0, self.ga.upper_value_limit) for x in range(self.ga.dimensions)]
        mutated = self.ga.mutate(sol, mutation_rate=1)
        for i, var in enumerate(mutated):
            self.assertNotEqual(sol[i], var)


class TestGACombinatorial(unittest.TestCase):
    def setUp(self) -> None:
        population_size = 10
        dimensions = 10
        self.ga = GA(dimensions, population_size, offspring_size=10, continuous_var_space=False, combinatorial_options=[20, 40, 60])

        @add_method(GA)
        def calc_fitness(variables):
            return sum(variables)

        self.ga.initialize_population()

    def test_swap_mutation(self):
        sol1 = random.choices(self.ga.combinatorial_values, k=self.ga.dimensions)
        self.ga.mutation_type = 'swap'
        mutated = self.ga.mutate(sol1, mutation_rate=1)
        self.assertNotEqual(sol1, mutated)

    def test_scramble_mutation(self):
        sol = random.choices(self.ga.combinatorial_values, k=self.ga.dimensions)
        self.ga.mutation_type = 'scramble'
        mutated = self.ga.mutate(sol, mutation_rate=1)
        self.assertNotEqual(sol, mutated)

    def test_bitflip_mutation(self):
        bitsol = random.choices([0, 1], k=self.ga.dimensions)
        self.ga.mutation_type = 'single bitflip'
        mutated = self.ga.mutate(bitsol, mutation_rate=1)
        self.assertNotEqual(bitsol, mutated)

    def test_multi_bitflip_mutation(self):
        bitsol2 = random.choices([0, 1], k=self.ga.dimensions)
        self.ga.mutation_type = 'multi bitflip'
        mutated = self.ga.mutate(bitsol2, mutation_rate=1)
        self.assertNotEqual(bitsol2, mutated)

    def test_create_offspring(self):
        children = self.ga.create_offspring()
        self.assertEqual(len(children), self.ga.offspring_size)


if __name__ == '__main__':
    unittest.main()
