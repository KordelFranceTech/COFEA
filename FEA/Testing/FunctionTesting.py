from numpy.random import seed, permutation
from numpy import dot, ones
from opfunu.cec.cec2010.utils import *


class F11_E:
    def __init__(self, shift_data_file="f11_op.txt", matrix_data_file="f11_m.txt"):
        self.matrix = load_matrix_data__(matrix_data_file)
        self.op_data = load_matrix_data__(shift_data_file)

    def run(self, solution=None, name="D/2m-group Shifted and m-rotated Ackley’s Function", m_group=50):
        problem_size = len(solution)
        epoch = int(problem_size / (2 * m_group))

        if problem_size == 1000:
            shift_data = self.op_data[:1, :].reshape(-1)
            permu_data = (self.op_data[1:, :].reshape(-1) - ones(problem_size)).astype(int)
        else:
            seed(0)
            shift_data = self.op_data[:1, :].reshape(-1)[:problem_size]
            permu_data = permutation(problem_size)
        z = solution - shift_data
        result = 0.0
        for i in range(0, epoch):
            idx1 = permu_data[i * m_group:(i + 1) * m_group]
            z1 = dot(z[idx1], self.matrix[:len(idx1), :len(idx1)])
            result += f4_ackley__(z1)
        idx2 = permu_data[int(problem_size / 2):problem_size]
        z2 = z[idx2]
        result += f4_ackley__(z2)
        return result

    def F11o(self, solution=None, name="D/2m-group Shifted and m-rotated Ackley’s Function", shift_data_file="f11_op.txt",
             matrix_data_file="f11_m.txt", m_group=50):
        problem_size = len(solution)
        epoch = int(problem_size / (2 * m_group))
        matrix = load_matrix_data__(matrix_data_file)
        op_data = load_matrix_data__(shift_data_file)
        if problem_size == 1000:
            shift_data = op_data[:1, :].reshape(-1)
            permu_data = (op_data[1:, :].reshape(-1) - ones(problem_size)).astype(int)
        else:
            seed(0)
            shift_data = op_data[:1, :].reshape(-1)[:problem_size]
            permu_data = permutation(problem_size)
        z = solution - shift_data
        result = 0.0
        for i in range(0, epoch):
            idx1 = permu_data[i * m_group:(i + 1) * m_group]
            z1 = dot(z[idx1], matrix[:len(idx1), :len(idx1)])
            result += f4_ackley__(z1)
        idx2 = permu_data[int(problem_size / 2):problem_size]
        z2 = z[idx2]
        result += f4_ackley__(z2)
        return result