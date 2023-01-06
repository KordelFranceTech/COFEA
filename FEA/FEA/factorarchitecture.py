from FEA.basealgorithms.ga import GA
from FEA.utilities.util import PopulationMember, add_method

import random
from tokenize import group
import numpy as np
import scipy as sp
import pandas as pd
from ast import literal_eval

try:
    import _pickle as pickle
except:
    import pickle
import os
# from utilities.clustering import FuzzyKmeans


def rotate(xs, n):
    return xs[n:] + xs[:n]


class FactorArchitecture(object):
    """
    Topology Generation:

    With d = 5, there are 5 variables, x_i => x_0, x_1, x_2, x_3, x_4. This means, in any list,
    the variable corresponds to the home. This is a convention and hopefully a space
    saving one.

    Of the four sort of topological elements required by the algorithm, there are two kinds:

    1. Lists where the indices implicitly refer to swarms and the values of the lists
       refer to variables or swarms.
    2. Lists where the indices implicitly refer to variables and the values of the lists
       refer to swarms.

    Factors and neighbors fall into the first group. The factor for swarm 0 is [0, 1].
    The neighbor of swarm 0 is swarm 1 (because they have variables in common).

    Arbiters fall into the second group as do optimizers. The arbiter of variable 0 is swarm 0.
    An arbiter of variable 3 is swarm 3 (0-3). Similarly, optimizers are lists of swarms
    that optimize for a specific home (variable). Variable 1 is optimized by [0, 1] which is
    to say both swarm 0 and swarm 1.

    factors = [[0, 1], [1, 2], [2, 3], [3, 4]] = # of factors and swarms
    neighbors =   [[1], [0, 2], [1, 3], [2]]
    arbiters  =   [0, 1, 2, 3, 3]
    optimizers =  [[0], [0, 1], [1, 2], [2, 3], [3]]
    """

    def __init__(self, dim=0, factors=None):
        """
        @param dim: number of dimensions, i.e., variables of a problem
        @param factors: if factors are generated externally, they can be initialized using this parameter
        """
        if factors is not None:
            self.factors = factors
            self.get_factor_topology_elements()
        else:
            self.factors = []
        self.arbiters = []
        self.optimizers = []
        self.neighbors = []
        self.dim = dim
        self.method = ""
        self.function_evaluations = 0

    def save_architecture(self, path_to_save=""):
        """
        Save generated factorarchitecture, i.e., "self" to a pickle fle
        @param path_to_save: string of static filepath with filename to save to (e.g. "C:/user/documents/factorarchitectures/my_factorarchitecture.pickle)
        """
        if path_to_save == "":
            if not os.path.isdir("factor_architecture_files/"):
                os.mkdir("factor_architecture_files/")
            if not os.path.isdir("factor_architecture_files/" + self.method):
                os.mkdir("factor_architecture_files/" + self.method)
            file = open("factor_architecture_files/" + self.method + "/" + self.method + "_" + str(self.dim), "wb")
        else:
            folder = os.path.dirname(path_to_save)
            if not os.path.isdir(folder):
                os.mkdir(folder)
            file = open(path_to_save, "wb")
        pickle.dump(self.__dict__, file)

    def load_architecture(self, path_to_load="", method="", dim=0):
        """
        Load architecture from pickle file.
        Can either send through full filepath, OR method name and dimensions to find existing file
        """
        from utilities.exceptions import PickleException
        if path_to_load == "" and (method == "" or dim == 0):
            raise PickleException()
        elif path_to_load != "" and os.path.isdir(path_to_load):
            raise PickleException()
        elif path_to_load == "" and method != "" and dim != 0:
            pickle_object = pickle.load(
                open("factor_architecture_files/" + method + "/" + method + "_" + str(dim), 'rb'))
            self.__dict__.update(pickle_object)
        elif path_to_load != "" and not os.path.isdir(path_to_load):
            pickle_object = pickle.load(open(path_to_load, 'rb'))
            self.__dict__.update(pickle_object)

    def load_csv_architecture(self, file_regex, dim, method=""):
        """
        Load architecture from csv file
        """
        frame = pd.read_csv(file_regex, header=0)
        frame.columns = map(str.upper, frame.columns)
        frame = frame.rename(columns={"DIM": "DIMENSION"}, errors="ignore")
        dim_frame = frame.loc[frame['DIMENSION'] == int(dim)]
        f = frame['FUNCTION'].unique()
        dim_array = np.array(dim_frame['FACTORS'])

        if epsilon == 0:
            home = dim_frame['NR_GROUPS'].argmax()
            factors = literal_eval(dim_array[home])
        else:
            epsilon_row = dim_frame.loc[dim_frame['EPSILON'] == epsilon]
            factors = literal_eval(np.array(epsilon_row['FACTORS'])[0])
        self.factors = factors
        self.dim = dim
        self.get_factor_topology_elements()


    def linear_grouping(self, group_size, offset):
        """
        create a linear grouping with specified group_size and offset, 
        e.g.: 10 variables, group_size = 5, offset = 3: factor 1 = [0,1,2,3,4], factor 2 = [3,4,5,6,7], factor 3 = [6,7,8,9]
        @param group_size: variables per factor
        @param offset: where to start the next factor group
        """
        self.method = "linear"
        assert offset <= group_size
        if offset == group_size:
            print("WARNING - offset and width are equal; the factors will not overlap.")
        self.factors = list(zip(*[range(i, self.dim, offset) for i in range(0, group_size)]))
        if self.factors[-1][-1] != self.dim-1:
            step_back = group_size - offset
            new_group = tuple(range(self.factors[-1][-step_back], self.dim, 1))
            self.factors.append(new_group)

    def ring_grouping(self, group_size=2):
        """
        create a ring architecture which rotates through the variables.
        """
        self.method = "ring"
        self.arbiters = list(range(0, self.dim))
        # self.factors = zip(*[rotate(self.arbiters, n) for n in range(0, group_size)])
        self.factors = [rotate(self.arbiters, n) for n in range(0, group_size)]
        print([rotate(self.arbiters, n) for n in range(0, group_size)])
        # print(self.factors)
        self.determine_neighbors()
        # print(self.neighbors)
        self.calculate_optimizers()
        # print(self.optimizers)

    def classic_random_grouping(self, group_size):
        """
        Random grouping as defined by Yang et al.
        Uses pre-defined group size to create distinct groupings, where variables are randomly added to groups.

        """
        self.method = "classic_random_"+str(group_size)
        number_of_groups = int(self.dim/group_size)
        indeces = list(range(0,self.dim))
        factors = []
        for n in range(number_of_groups-1):
            grp = random.sample(indeces, k=group_size)
            factors.append(grp)
            for grpidx in grp:
                indeces.remove(grpidx)
        factors.append(indeces)
        self.factors = factors

    def random_grouping(self, min_groups=5, max_groups=15, overlap=False):
        self.method = "random"
        number_of_groups = random.randint(min_groups,max_groups)
        while True:
            # make sure each group has at least one variable
            assigned_groups = random.choices(range(0,number_of_groups), k=self.dim)
            if len(set(assigned_groups)) == number_of_groups:
                break
        factors = []
        for i in range(number_of_groups):
            # check which variables belong to each group and add them to the factor group
            factor = [idx for idx, grp in enumerate(assigned_groups) if grp == i]
            factors.append(factor)
        if overlap:
            for i in range(0,number_of_groups-1):
                factor = random.sample(self.factors[i],k=int(np.ceil(len(self.factors[i])/4)))
                factor.extend(random.sample(self.factors[i+1],k=int(np.ceil(len(self.factors[i+1])/4))))
                factors.append(factor)
        return factors

    def genetic_grouping(self, problem, population_size=200, ga_runs=100, c1=0, c2=1):
        """
        Genetic grouping strategy to group variables in combinatorial problems without pre-defining group sizes.
        """
        # initialize GA to perform optimization
        ga = GA(dimensions=self.dim, population_size=population_size, mutation_type="grouping", crossover_type="grouping", parent_selection="grouping")
        # define fitness function to check difference between full evaluation and group evaluations
        full_c1 = np.ones(self.dim)*c1
        full_c2 = np.ones(self.dim)*c2
        full_c1_fitness = np.sum(problem.evaluate(full_c1))
        full_c2_fitness = np.sum(problem.evaluate(full_c2))
        
        def calc_fitness(variables):
            # variables here are the genes in the chromosome, which can have different lengths
            chromosome_length = len(variables)
            full_fitness = chromosome_length * (full_c1_fitness+full_c2_fitness)
            print("fitness for c1/c2 combined", full_fitness)
            per_group_fitness = 0
            for gene in variables:
                group_solution_c1 = np.ones(self.dim)*c2
                group_solution_c2 = np.ones(self.dim)*c1
                for var_idx in gene:
                    group_solution_c1[var_idx] = c1
                    group_solution_c2[var_idx] = c2
                per_group_fitness += np.sum(problem.evaluate(group_solution_c1))
                print("c1 group added: ", per_group_fitness)
                per_group_fitness += np.sum(problem.evaluate(group_solution_c2))
                print("c2 group added: ", per_group_fitness)
            return abs(full_fitness-per_group_fitness)
                
        # generate population: each chromosome consists of genes representing a group of variables
        population = []
        for i in range(population_size):
            chromosome = self.random_grouping(min_groups=1, max_groups=self.dim)
            chromosome_fitness = calc_fitness(chromosome)
            print(chromosome, chromosome_fitness)
            population.append(PopulationMember(chromosome, chromosome_fitness))
        run = 0
        VI_diff = np.inf
        # until max gen or varinteraction_diff != 0
        while run < ga_runs and VI_diff != 0:
            children = ga.create_offspring(population)
            offspring = []
            for child in children:
                offspring.append(PopulationMember(child, calc_fitness(child)))
            total_population = population + offspring
            population = ga.selection(total_population)
            VI_diff = population[0].fitness
            run += 1
        print(population[0])
        self.factors = population[0].variables

    def diff_grouping(self, _function, epsilon, m=0, moo=False, n_obj=np.inf):
        """
        DIFFERENTIAL GROUPING
        Omidvar et al. 2010
        """
        self.method = "DG"
        size = self.dim
        dimensions = np.arange(start=0, stop=size)
        curr_dim_idx = 0
        factors = []
        separate_variables = []
        function_evaluations = 0
        loop = 0

        while size > 0:
            # initialize for current iteration
            curr_factor = [dimensions[0]]

            curr_factor = self.check_delta(_function, m, 1, size, dimensions, epsilon, curr_factor, moo, n_obj)

            if len(curr_factor) == 1:
                separate_variables.extend(curr_factor)
            else:
                factors.append(tuple(curr_factor))

            # Final adjustments
            indices_to_delete = np.searchsorted(dimensions, curr_factor)
            dimensions = np.delete(dimensions, indices_to_delete)  # remove j from dimensions
            size = len(dimensions)
            if size != 0:
                curr_dim_idx = dimensions[0]

            loop += 1
        if len(separate_variables) != 0:
            factors.append(tuple(separate_variables))

        self.factors = factors

    def overlapping_diff_grouping(self, _function, epsilon, m=0, moo=False, n_obj=np.inf):
        """
        Use differential grouping approach to determine overlapping factors.
        :return:
        """
        self.method = "ODG"
        size = self.dim
        dimensions = np.arange(start=0, stop=size)
        factors = []
        separate_variables = []
        function_evaluations = 0
        loop = 0

        for i, dim in enumerate(dimensions):
            # initialize for current iteration
            curr_factor = [dim]

            self.check_delta(_function, m, i, size, dimensions, epsilon, curr_factor, moo, n_obj)

            if len(curr_factor) == 1:
                separate_variables.extend(curr_factor)
            else:
                factors.append(tuple(curr_factor))

            loop += 1

        factors.append(tuple(separate_variables))
        self.factors = factors

    def check_delta(self, _function, m, i, size, dimensions, eps, curr_factor, moo=False, n_obj=np.inf):
        """
        Helper function for the two differential grouping approaches.
        Compares function fitnesses to determine whether there is a difference in results larger than 'epsilon'.
        :param _function:
        :param m:
        :param i:
        :param size:
        :param dimensions:
        :param eps:
        :param curr_factor:
        :return curr_factor:
        """
        p1 = [random.random() for x in range(self.dim)]
        p2 = [x for x in p1]
        p2[i] = p1[i]+.1
        if not moo:
            if m == 0:
                delta1 = _function(p1) - _function(p2)
            else:
                delta1 = _function(p1, m_group=m) - _function(p2, m_group=m)
        elif moo:
            delta1 = _function.evaluate(p1)[n_obj] - _function.evaluate(p2)[n_obj]
            #print("obj: ", n_obj, "D1: ", delta1)
        self.function_evaluations += 2

        for j in range(i + 1, size):
            p3 = [x for x in p1]
            p4 = [x for x in p2]
            rand = random.random()
            p3[j] = rand
            p4[j] = rand

            if not moo:
                if m == 0:
                    delta2 = _function(p3) - _function(p4)
                else:
                    delta2 = _function.run(p3, m_group=m) - _function.run(p4, m_group=m)
            elif moo:
                delta2 = _function.evaluate(p3)[n_obj] - _function.evaluate(p4)[n_obj]
                #print("obj: ", n_obj, "D2: ", delta2, _function.evaluate(p3)[n_obj], _function.evaluate(p4)[n_obj])
            self.function_evaluations += 2

            if abs(delta1 - delta2) > eps:
                curr_factor.append(dimensions[j])

        return curr_factor

    # def spectral_grouping(self, IM, num_clusters):
    #     from networkx import to_networkx_graph, Graph
    #     from networkx.linalg import laplacian_matrix
    #     '''
    #     Assign the datapoints to clusters using spectral clustering and return and array of cluster assignemnts
    #     '''
    #     self.method = "spectral"
    #     IM_graph = to_networkx_graph(IM, create_using=Graph)
    #
    #     # get Laplacian
    #     laplacian = sp.sparse.csr_matrix.toarray(laplacian_matrix(IM_graph))
    #
    #     # calc eigen vectors and values of the laplacian
    #     eig_values, eig_vectors = np.linalg.eig(laplacian)
    #     sorted_indices = eig_values.argsort()
    #     eig_values = eig_values[sorted_indices]
    #     eig_vectors = eig_vectors[sorted_indices]
    #
    #     # take k largest eigen vectors
    #     k_arr = np.arange(num_clusters)
    #     eig_values = eig_values[k_arr]
    #     eig_vectors = np.transpose(eig_vectors[k_arr])
    #
    #     # run fuzzy kmeans with the eigen vectors
    #     self.factors = FuzzyKmeans(eig_vectors, num_clusters).assign_clusters()

    def MEET(self, T):
        """
        Create directed graph with edge weights in MIC table.
        Directed graph (IM) can be calculated using different methods, called from variableinteraction class
        Create MAXimal spanning tree from this graph.
        :param T, T is either a directed graph stored as a numpy array, or a networkx graph object
        :return:
        """

        if isinstance(T, np.ndarray):  # convert np array to tree
            from networkx import  from_numpy_array, maximum_spanning_tree
            G = from_numpy_array(T)
            T = maximum_spanning_tree(G)

        self.method = "MEET"
        factors = []

        print(f'Total weight: {T.size(weight="weight")}')

        for node in list(T.nodes):  # each dimension
            factor = list(T.neighbors(node))  # adjacent nodes
            factor.append(node)  # add itself to the group
            factors.append(factor)

        self.factors = factors

    def MEET2(self, T, number_of_factors=50):
        """
        Create directed graph with edge weights in MIC table.
        Directed graph (IM) can be calculated using different methods, called from variableinteraction class
        Create MAXimal spanning tree from this graph.
        :param T, T is either a directed graph stored as a numpy array, or a networkx graph object
        :return:
        """

        if isinstance(T, np.ndarray):  # convert np array to tree
            from networkx import  from_numpy_array, maximum_spanning_tree
            G = from_numpy_array(T)
            T = maximum_spanning_tree(G)

        self.method = "MEET"
        factors = []

        print(f'Total weight: {T.size(weight="weight")}')

        for node in list(T.nodes):  # each dimension
            factor = list(T.neighbors(node))  # adjacent nodes
            factor.append(node)  # add itself to the group
            factors.append(factor)

        while len(factors) > number_of_factors:  # shrink the number of factors!
            factors.sort(key=len)
            f1 = factors.pop(0)
            f2 = factors.pop(0)
            new_f = set(f1 + f2)
            factors.append(list(new_f))

        self.factors = factors

    def get_factor_topology_elements(self):
        """
        Calculates arbiters, optimizers and neighbours based on the created factors.
        """
        self.nominate_arbiters()
        self.calculate_optimizers()
        self.determine_neighbors()

    def nominate_arbiters(self):
        """
        The arbiter of variable 0 is swarm 0.
        An arbiter of variable 3 is swarm 3 (0-3).
        :return:
        """
        assignments = {}
        # Iteration is faster when it does not have to access the object each time
        factors = [f for f in self.factors]
        for i, factor in enumerate(factors[:-1]):
            for j in factor:
                if j not in self.factors[i + 1] and j not in assignments:
                    assignments[j] = i
        for j in factors[-1]:
            if j not in assignments:
                assignments[j] = len(factors) - 1
        keys = list(assignments.keys())
        keys.sort()
        arbiters = [assignments[k] for k in keys]
        self.arbiters = arbiters

    def calculate_optimizers(self):
        """
        Optimizers are lists of swarms that optimize for a specific home (variable).
        Variable 1 is optimized by swarms [0, 1].
        :return:
        """
        optimizers = []
        factors = [f for f in self.factors]
        for v in range(self.dim):
            optimizer = []
            for i, factor in enumerate(factors):
                if v in factor:
                    optimizer.append(i)
            optimizers.append(optimizer)
        self.optimizers = optimizers

    def determine_neighbors(self):
        """
        The factor for swarm 0 is [0, 1].
        The neighbor of swarm 0 is swarm 1 (because they have variables in common).
        :return:
        """
        neighbors = []
        factors = [f for f in self.factors]
        for i, factor in enumerate(factors):
            neighbor = []
            for j, other_factor in enumerate(factors):
                if (i != j) and not set(factor).isdisjoint(set(other_factor)):
                    neighbor.append(j)
            neighbors.append(neighbor)
        self.neighbors = neighbors


from FEA.utilities.util import compare_solutions
from networkx import from_numpy_array, connected_components


class MooFactorArchitecture:
    """
    Create factor architecture along different objective axes.
    Need to be generalized a bit more.
    """
    def __init__(self, dim, n_obj, problem=None, decomp_approach='diff_grouping'):
        self.dim = dim
        self.problem = problem
        self.n_obj = n_obj
        self.decomp = decomp_approach
        self.n_obj = n_obj

    def graph_based_MOO_dg(self, ubound=1, lbound=0, shift_param=111.1, nr_samples=20):
        if isinstance(ubound, int):
            ubound_vars = np.ones(self.dim) * ubound
            lbound_vars = np.ones(self.dim) * lbound
        else:
            ubound_vars = ubound
            lbound_vars = lbound
        # Property analysis
        diversity_variables, convergence_variables = self.variable_property_analysis(ubound_vars, lbound_vars, shift_param, nr_samples)
        # Interaction learning
        interaction_matrix = self.graph_DG_interaction_learning(convergence_variables, ubound_vars, lbound_vars, shift_param, omega=1e-08)
        # Graph grouping
        G = from_numpy_array(interaction_matrix)
        components = [list(g) for g in connected_components(G)]
        fa = FactorArchitecture(self.dim, factors=components)
        return fa

    def variable_property_analysis(self, ubound, lbound, shift_param=111.1, nr_samples=20):
        diversity_variables = []
        convergence_variables = []
        for i in range(self.dim):
            solutions = []
            for j in range(nr_samples):
                solution = []
                shift_value = j-1 + (1/shift_param)
                for k in range(self.dim):
                    if i == k:
                        solution.append((shift_value/nr_samples) * (ubound[k]-lbound[k]) + lbound[k])
                    else:
                        solution.append(.5 * (ubound[k]+lbound[k]) + shift_value)
                solutions.append(solution)
            flag = False
            for j in range(nr_samples):
                for k in range(j+1, nr_samples):
                    if compare_solutions(solutions[j], solutions[k]) == 0:
                        flag = True
            if flag:
                diversity_variables.append(i)
            else:
                convergence_variables.append(i)
        return diversity_variables, convergence_variables

    def graph_DG_interaction_learning(self, convergence_vars, ubound, lbound, shift_param, omega):
        max_weights = np.zeros(self.n_obj)
        min_weights = np.ones(self.n_obj) * 10000
        fitnesses = []
        fitness_matrix = np.zeros((len(convergence_vars), len(convergence_vars), self.n_obj))
        shift_vector = (ubound - lbound) / shift_param
        x0 = lbound+shift_vector
        f0 = self.problem.evaluate(x0)
        for i, convergence_idx in enumerate(convergence_vars):
            x1 = [x for x in x0]
            x1[convergence_idx] = ubound[convergence_idx] - shift_vector[convergence_idx]
            fitnesses.append(self.problem.evaluate(x1))
            for j in range(i+1, len(convergence_vars)):
                x2 = [x for x in x1]
                x2[convergence_vars[j]] = ubound[convergence_vars[j]] - shift_vector[convergence_vars[j]]
                fitness_matrix[i, j] = self.problem.evaluate(x2)
        delta_matrix = np.zeros((self.n_obj, len(convergence_vars), len(convergence_vars)))
        for i, convergence_idx in enumerate(convergence_vars):
            for j in range(i + 1, len(convergence_vars)):
                for l in range(self.n_obj):
                    delta1 = f0[l] - fitnesses[i][l]
                    delta2 = fitnesses[j][l] - fitness_matrix[i, j][l]
                    diff = abs(delta1 - delta2)
                    delta_matrix[l, i, j] = diff
                    delta_matrix[l, j, i] = diff
                    if min_weights[l] > diff:
                        min_weights[l] = diff
                    if max_weights[l] < diff:
                        max_weights[l] = diff
        IM = np.zeros((len(convergence_vars), len(convergence_vars)))
        for i, convergence_idx in enumerate(convergence_vars):
            for j in range(i + 1, len(convergence_vars)):
                for l in range(self.n_obj):
                    # perform normalization if weights are larger than threshold omega
                    if (max_weights[l]-min_weights[l]) > omega:
                        delta_matrix[l,i,j] = min_weights[l] + ((delta_matrix[l,i,j]-min_weights[l])/(max_weights[l]-min_weights[l]))
                    if delta_matrix[l,i,j] > omega:
                        IM[i, j] = 1
                        IM[j, i] = 1
        return IM

    def create_objective_factors(self, save_files=False, disjoint=False) -> FactorArchitecture:
        """Create factors along different objective functions.
        For each objective, a FactorArchitecture object is created.
        :param save_files: Boolean that determines whether the created factorArchitectures are saved in pickle files
        :returns FactorArchitecture object: with all the factors generated
        """
        all_factors = FactorArchitecture(self.dim)
        all_factors.method = self.decomp + '_MOO'
        eps = 3
        for i in range(self.n_obj):
            fa = FactorArchitecture(self.dim)
            if self.decomp == 'diff_grouping':
                getattr(fa, self.decomp)(self.problem, eps, moo=True, n_obj=i)
            if save_files:
                fa.save_architecture(
                    '../factor_architecture_files/n_obj_' + str(self.n_obj) + '_' + fa.method + '_dim_' + str(self.dim) + '_obj_' + str(i))
            print(fa.factors)
            print(len(fa.factors))
            for f in fa.factors:
                all_factors.factors.append(f)
        # if disjoint:

        all_factors.get_factor_topology_elements()
        return all_factors

    def read_objective_factors(self, method_name) -> FactorArchitecture:
        all_factors = FactorArchitecture(self.dim)
        for i, obj in enumerate(self.n_obj):
            fa = FactorArchitecture(self.dim)
            fa.load_architecture(
                '../factor_architecture_files/n_obj_' + str(self.n_obj) + '_' + method_name + '_dim_' + str(self.dim) + '_obj_' + str(i))
            all_factors.factors.extend(fa.factors)
        all_factors.get_factor_topology_elements()
        return all_factors

if __name__ == "__main__":
    from pymoo.factory import get_problem
    import re

    current_working_dir = os.getcwd()
    path = re.search(r'^(.*?[\\/]FEA)', current_working_dir)
    path = path.group()

    function_name = 'DTLZ1'
    n_obj=5
    problem = get_problem(function_name, n_var=1000, n_obj=n_obj)
    moofa = MooFactorArchitecture(dim=1000, problem=problem, n_obj=n_obj)
    #factors = moofa.graph_based_MOO_dg()
    factors = moofa.create_objective_factors()
    # print(len(factors.factors))
    factors.save_architecture(path_to_save=path+"/FEA/factor_architecture_files/DG_MOO/DG_"+function_name+"_"+str(n_obj))