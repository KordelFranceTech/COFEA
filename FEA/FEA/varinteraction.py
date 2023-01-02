import numpy as np
from minepy import MINE
from networkx import from_numpy_array, maximum_spanning_tree, connected_components
import random


class MEE(object):
    def __init__(self, func, dim, samples, mic_thresh, de_thresh, delta, measure = None, use_mic_value=True):
        self.f = func
        self.d = dim
        rand = random.random()
        self.ub = np.ones(self.d) * rand
        self.lb = np.ones(self.d) * rand
        self.samples = samples
        self.mic_thresh = mic_thresh  # mic threshold
        self.de_thresh = de_thresh  # diff equation (de) threshold
        self.delta = delta  # account for small variations
        self.IM = np.zeros((self.d, self.d))
        self.use_mic_value = use_mic_value
        self.measure = measure
        # Define measure
        # self.measure = Entropic(self.f, self.d, self.lb, self.ub, self.samples, self.delta, self.de_thresh)
        #self.measure = DGInteraction(self.f, self.d, 0.001, m=4)

    def get_IM(self):
        self.direct_IM()
        if not self.use_mic_value:
            self.strongly_connected_comps()
        return self.IM

    def direct_IM(self):

        """
        Calculates the Direct Interaction Matrix based on MIC
        :return: direct_IM
        """
        f, dim, lb, ub, sample_size, delta = self.f, self.d, self.lb, self.ub, self.samples, self.delta
        # for each dimension
        for i in range(dim):
            print("dim: ", i)
            # compare to consecutive variable (/dimension)
            for j in range(i + 1, dim):

                mic = self.measure.compute(i, j)

                if self.use_mic_value:
                    # print(mic, end=", ")
                    self.IM[i, j] = mic
                elif not self.use_mic_value and mic > self.mic_thresh:  # threshold <--------
                    self.IM[i, j] = 1
                    self.IM[j, i] = 1

    def strongly_connected_comps(self):
        from networkx import to_networkx_graph, DiGraph
        from networkx.algorithms.components import strongly_connected_components

        """
        Sets strongly connected components in the Interaction Matrix
        """
        IM_graph = to_networkx_graph(self.IM, create_using= DiGraph)
        strongly_connected_components = strongly_connected_components(IM_graph)
        for component in strongly_connected_components:
            component = list(component)
            for i in range(len(component)):
                for j in range(i + 1, len(component)):
                    self.IM[component[i], component[j]] = 1
                    self.IM[component[j], component[i]] = 1


class RandomTree(object):
    def __init__(self, dim, measure):
        self.d = dim
        r = np.random.random(size=(self.d, self.d))

        self.IM = (r + r.T) - 2  # init IM to be symmetric with random values between [-2, 0]
        # self.IM = np.ones((self.d, self.d)) * -1  # init IM to bunch of -1's (so we can initialize a tree)

        self.iteration_ctr = 0

        # Init tree and graph
        self.G = from_numpy_array(self.IM)  # We don't technically need this in self, but might as well have it
        self.T = maximum_spanning_tree(self.G)  # just make a tree (they're all -1 so it is a boring tree)

        self.measure = measure

    def run(self, trials):
        """
        Runs a greedy improvement algorithm on the existing tree
        Replaces the cheapest edge, and adds a new edge if it is less expensive then the queried edge
        :param trials: The number of iterations to run
        :return:
        """
        summary = ""
        for i in range(trials):
            self.iteration_ctr += 1  # keep track of global counter to allow for multiple, sequential run calls

            edges = list(self.T.edges(data="weight"))
            remove = random.choice(edges)  # remove a random edge
            # remove = min(edges, key=lambda e: e[2])  # find the cheapest edge
            self.T.remove_edge(remove[0], remove[1])  # delete the edge

            comp1, comp2 = connected_components(self.T)

            node1 = random.choice(list(comp1))  # generate random start node
            node2 = random.choice(list(comp2))  # generate random end node

            interact = self.compute_interaction(node1, node2)
            summary += f"\t|\t{remove[2]} --> {interact} "
            if interact > remove[2]:  # if the new random edge is more expensive then the previous one, add it
                self.T.add_edge(node1, node2, weight=interact)
                summary += "Accepted"
            else:  # otherwise add the original one back
                self.T.add_edge(remove[0], remove[1], weight=remove[2])
                summary += "Rejected"
        print(summary)
        return self.T

    def compute_interaction(self, i, j):
        """
        Computes the interaction using MEE between vars i, j
        :param i:
        :param j:
        :return: MIC value
        """
        if self.IM[i][j] > 0:
            print('IM larger')
            return self.IM[i][j]

        mic = self.measure.compute(i, j)

        self.IM[i, j] = mic
        self.IM[j, i] = mic
        return mic

class Measure(object):
    """
    Base class
    """

    def compute(self, i, j):
        return 0


class Entropic(Measure):
    def __init__(self, f, d, samples, delta, de_thresh,moo=False, n_obj=0):
        """
        Uses MEE to compute interaction
        :param f: function
        :param d: dimensions
        :param samples: number of samples to take
        :param delta: pertubation
        :param de_thresh: threshold value
        """
        self.de_thresh = de_thresh
        self.delta = delta
        self.samples = samples
        self.d = d
        self.f = f
        rand = random.random()
        self.ub = np.ones(self.d) * rand
        self.lb = np.ones(self.d) * (rand-.25)
        self.moo = moo
        self.n_obj = n_obj

    def compute(self, i, j):
        # number of values to calculate == sample size
        f, dim, sample_size, delta = self.f, self.d, self.samples, self.delta
        de = np.zeros(sample_size)
        # generate n values (i.e. samples) for j-th dimension
        x_j = np.random.rand(sample_size) * (self.ub[j] - self.lb[j]) + self.lb[j]
        # randomly generate solution -- initialization of function variables
        x = np.random.uniform(self.lb, self.ub, size=dim)
        for k in range(1, sample_size):
            cp = x[j]
            x[j] = x_j[k]  # set jth value to random sample value
            if not self.moo:
                y_1 = f.run(x)
            else:
                y_1 = f.evaluate(x)[self.n_obj]
            x[i] = x[i] + delta
            if not self.moo:
                y_2 = f.run(x)
            else:
                y_2 = f.evaluate(x)[self.n_obj]
            de[k] = (y_2 - y_1) / delta
            # Reset the changes
            x[j] = cp
            x[i] = x[i] - delta

        avg_de = np.mean(de)
        de[de < self.de_thresh] = avg_de  # use np fancy indexing to replace values

        mine = MINE()
        mine.compute_score(de, x_j)
        mic = mine.mic()
        return mic


class DGInteraction(Measure):
    def __init__(self, func, dim, epsilon, n_obj=0, moo=False, m=0):
        self.f = func
        self.dim = dim
        self.eps = epsilon
        self.m = m
        self.n_obj = n_obj
        self.moo = moo

    def compute(self, i, j):
        p1 = [random.random() for x in range(self.dim)]
        p2 = [x for x in p1]
        while True:
            rand1 = random.random()
            if abs(rand1 - p1[i]) > 0.1:
                break
        while True:
            rand2 = random.random()
            if abs(rand2 - p1[j]) > 0.1:
                break
        p2[i] = rand1
        if not self.moo:
            delta1 = self.f.run(p1) - self.f.run(p2)
        else:
            delta1 = self.f.evaluate(p1)[self.n_obj] - self.f.evaluate(p2)[self.n_obj]
            # print("obj: ", self.n_obj, "D1: ", delta1)

        p1[j] = rand2
        p2[j] = rand2

        if not self.moo:
            delta2 = self.f.run(p1) - self.f.run(p2)
        else:
            delta2 = self.f.evaluate(p1)[self.n_obj] - self.f.evaluate(p2)[self.n_obj]
            # print("obj: ", self.n_obj, "D2: ", delta2)

        return abs(delta1 - delta2)


if __name__ == '__main__':
    pass
    #f = Function(function_number=1, shift_data_file="f01_o.txt")
    #mee = MEE(f, 5, 5, 0.1, 0.0001, 0.000001)
    #MEE(func=f, dim=5, samples=5, mic_thresh=0.1, de_thresh=0.0001, delta=0.000001)
    #mee.get_IM()