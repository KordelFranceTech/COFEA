#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 12:29, 20/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#
# Modified by Elliott Pryor 08 March 2021 & Amy Peerlinck Apr 16 2021
#-------------------------------------------------------------------------------------------------------%

from numpy.random import seed, permutation
from numpy import dot, ones
from opfunu.cec.cec2010.utils import *
from optimizationproblems.benchmarks import *


class Function(object):

    def __init__(self, function_number=0, partial_function=None, lbound=-100, ubound=100, shift_data_file="", matrix_data_file="", m=0):
        self.function_to_call = 'F'+str(function_number)
        self.name = ""
        self.dimensions = 0
        self.lbound = lbound
        self.ubound = ubound
        self.m_shift = m
        self.shift_data = None
        self.matrix_data = None

        if shift_data_file != "" and matrix_data_file == "":
            if 4 > function_number or (18 < function_number < 21):
                self.shift_data = load_shift_data__(shift_data_file)
            else:
                self.shift_data = load_matrix_data__(shift_data_file)
        elif matrix_data_file != "":
            self.matrix_data = load_matrix_data__(matrix_data_file)
            self.shift_data = load_matrix_data__(shift_data_file)

    def run(self, solution):
        if self.dimensions == 0:
            self.dimensions = len(solution)
            # check_problem_size(self.dimensions)
        return getattr(self, self.function_to_call)(solution=solution)

    def shift_permutation(self):
        if self.dimensions == 1000:
            shift_data = self.shift_data[:1, :].reshape(-1)
            permu_data = (self.shift_data[1:, :].reshape(-1) - ones(self.dimensions)).astype(int)
        else:
            seed(0)
            shift_data = self.shift_data[:1,:].reshape(-1)[:self.dimensions]
            permu_data = permutation(self.dimensions)
        return shift_data, permu_data

    def F1(self, solution=None, name="Shifted Elliptic Function"):
        self.name = name
        z = solution - self.shift_data[:self.dimensions]
        return elliptic__(z)

    def F2(self, solution=None, name="Shifted Rastrigin’s Function"):
        self.name = name
        z = solution - self.shift_data[:self.dimensions]
        return rastrigin__(z)

    def F3(self, solution=None, name="Shifted Ackley’s Function"):
        self.name = name
        z = solution - self.shift_data[:self.dimensions]
        return ackley__(z)

    def F4(self, solution=None, name="Single-group Shifted and m-rotated Elliptic Function", m_group=50):
        self.name = name
        shift_data, permu_data = self.shift_permutation()
        z = solution - shift_data
        idx1 = permu_data[:m_group]
        idx2 = permu_data[m_group:]
        z_rot_elliptic = dot(z[idx1], self.matrix_data[:m_group, :m_group])
        z_elliptic = z[idx2]
        return elliptic__(z_rot_elliptic) * 10**6 + elliptic__(z_elliptic)
    
    def F5(self, solution=None, name="Single-group Shifted and m-rotated Rastrigin’s Function", m_group=50):
        self.name = name
        shift_data, permu_data = self.shift_permutation()
        z = solution - shift_data
        idx1 = permu_data[:m_group]
        idx2 = permu_data[m_group:]
        z_rot_rastrigin = dot(z[idx1], self.matrix_data[:m_group, :m_group])
        z_rastrigin = z[idx2]
        return rastrigin__(z_rot_rastrigin) * 10 ** 6 + rastrigin__(z_rastrigin)
    
    def F6(self, solution=None, name="Single-group Shifted and m-rotated Ackley’s Function", m_group=50):
        self.name = name
        shift_data, permu_data = self.shift_permutation()
        z = solution - shift_data
        idx1 = permu_data[:m_group]
        idx2 = permu_data[m_group:]
        z_rot_ackley = dot(z[idx1], self.matrix_data[:m_group, :m_group])
        z_ackley = z[idx2]
        return ackley__(z_rot_ackley) * 10 ** 6 + ackley__(z_ackley)
       
    def F7(self, solution=None, name="Single-group Shifted m-dimensional Schwefel’s Problem 1.2", m_group=50):
        self.name = name
        shift_data, permu_data = self.shift_permutation()
        z = solution - shift_data
        idx1 = permu_data[:m_group]
        idx2 = permu_data[m_group:]
        z_schwefel = z[idx1]
        z_shpere = z[idx2]
        return schwefel__(z_schwefel) * 10 ** 6 + sphere__(z_shpere)
      
    def F8(self, solution=None, name=" Single-group Shifted m-dimensional Rosenbrock’s Function", m_group=50):
        self.name = name
        shift_data, permu_data = self.shift_permutation()
        z = solution - shift_data
        idx1 = permu_data[:m_group]
        idx2 = permu_data[m_group:]
        z_rosenbrock = z[idx1]
        z_sphere = z[idx2]
        return rosenbrock__(z_rosenbrock) * 10 ** 6 + sphere__(z_sphere)
      
    def F9(self, solution=None, name="D/2m-group Shifted and m-rotated Elliptic Function", m_group=50):
        self.name = name
        epoch = int(self.dimensions / (2 * m_group))
        # check_m_group("F9", self.dimensions, 2*m_group)
        shift_data, permu_data = self.shift_permutation()
        z = solution - shift_data
        result = 0.0
        for i in range(0, epoch):
            idx1 = permu_data[i*m_group:(i+1)*m_group]
            z1 = dot(z[idx1], self.matrix_data[:len(idx1), :len(idx1)])
            result += elliptic__(z1)
        idx2 = permu_data[int(self.dimensions/2):self.dimensions]
        z2 = z[idx2]
        result += elliptic__(z2)
        return result
       
    def F10(self, solution=None, name="D/2m-group Shifted and m-rotated Rastrigin’s Function", m_group=50):
        self.name = name
        epoch = int(self.dimensions / (2 * m_group))
        # check_m_group("F10", self.dimensions, 2*m_group)
        shift_data, permu_data = self.shift_permutation()
        z = solution - shift_data
        result = 0.0
        for i in range(0, epoch):
            idx1 = permu_data[i * m_group:(i + 1) * m_group]
            z1 = dot(z[idx1], self.matrix_data[:len(idx1), :len(idx1)])
            result += rastrigin__(z1)
        idx2 = permu_data[int(self.dimensions / 2):self.dimensions]
        z2 = z[idx2]
        result += rastrigin__(z2)
        return result
     
    def F11(self, solution=None, name="D/2m-group Shifted and m-rotated Ackley’s Function", m_group=50):
        self.name = name
        epoch = int(self.dimensions / (2 * m_group))
        # check_m_group("F11", self.dimensions, 2*m_group)
        shift_data, permu_data = self.shift_permutation()
        z = solution - shift_data
        result = 0.0
        for i in range(0, epoch):
            idx1 = permu_data[i * m_group:(i + 1) * m_group]
            z1 = dot(z[idx1], self.matrix_data[:len(idx1), :len(idx1)])
            result += ackley__(z1)
        idx2 = permu_data[int(self.dimensions / 2):self.dimensions]
        z2 = z[idx2]
        result += ackley__(z2)
        return result
      
    def F12(self, solution=None, name="D/2m-group Shifted m-dimensional Schwefel’s Problem 1.2", m_group=50):
        self.name = name
        epoch = int(self.dimensions / (2 * m_group))
        # check_m_group("F12", self.dimensions, 2*m_group)
        shift_data, permu_data = self.shift_permutation()
        z = solution - shift_data
        result = 0.0
        for i in range(0, epoch):
            idx1 = permu_data[i * m_group:(i + 1) * m_group]
            result += schwefel__(z[idx1])
        idx2 = permu_data[int(self.dimensions / 2):self.dimensions]
        result += sphere__(z[idx2])
        return result
      
    def F13(self, solution=None, name="D/2m-group Shifted m-dimensional Rosenbrock’s Function", m_group=50):
        self.name = name
        epoch = int(self.dimensions / (2 * m_group))
        # check_m_group("F13", self.dimensions, 2*m_group)
        shift_data, permu_data = self.shift_permutation()
        z = solution - shift_data
        result = 0.0
        for i in range(0, epoch):
            idx1 = permu_data[i * m_group:(i + 1) * m_group]
            result += rosenbrock__(z[idx1])
        idx2 = permu_data[int(self.dimensions / 2):self.dimensions]
        result += sphere__(z[idx2])
        return result
     
    def F14(self, solution=None, name="D/2m-group Shifted and m-rotated Elliptic Function", m_group=50):
        self.name = name
        epoch = int(self.dimensions / m_group)
        # check_m_group("F14", self.dimensions, m_group)
        shift_data, permu_data = self.shift_permutation()
        z = solution - shift_data
        result = 0.0
        for i in range(0, epoch):
            idx1 = permu_data[i * m_group:(i + 1) * m_group]
            result += elliptic__(dot(z[idx1], self.matrix_data))
        return result
      
    def F15(self, solution=None, name="D/2m-group Shifted and m-rotated Rastrigin’s Function", m_group=50):
        self.name = name
        epoch = int(self.dimensions / m_group)
        # check_m_group("F15", self.dimensions, m_group)
        shift_data, permu_data = self.shift_permutation()
        z = solution - shift_data
        result = 0.0
        for i in range(0, epoch):
            idx1 = permu_data[i * m_group:(i + 1) * m_group]
            result += rastrigin__(dot(z[idx1], self.matrix_data))
        return result
  
    def F16(self, solution=None, name="D/2m-group Shifted and m-rotated Ackley’s Function", m_group=50):
        self.name = name
        epoch = int(self.dimensions / m_group)
        # check_m_group("F16", self.dimensions, m_group)
        shift_data, permu_data = self.shift_permutation()
        z = solution - shift_data
        result = 0.0
        for i in range(0, epoch):
            idx1 = permu_data[i * m_group:(i + 1) * m_group]
            result += ackley__(dot(z[idx1], self.matrix_data))
        return result
  
    def F17(self, solution=None, name="D/2m-group Shifted m-dimensional Schwefel’s Problem 1.2", m_group=4):
        self.name = name
        epoch = int(self.dimensions / m_group)
        # check_m_group("F17", self.dimensions, m_group)
        shift_data, permu_data = self.shift_permutation()
        z = solution - shift_data
        result = 0.0
        for i in range(0, epoch):
            idx1 = permu_data[i * m_group:(i + 1) * m_group]
            result += schwefel__(z[idx1])
        return result

    def F18(self, solution=None, name="D/2m-group Shifted m-dimensional Rosenbrock’s Function", m_group=50):
        self.name = name
        epoch = int(self.dimensions / m_group)
        # check_m_group("F18", self.dimensions, m_group)
        shift_data, permu_data = self.shift_permutation()
        z = solution - shift_data
        result = 0.0
        for i in range(0, epoch):
            idx1 = permu_data[i * m_group:(i + 1) * m_group]
            result += rosenbrock__(z[idx1])
        return result
   
    def F19(self, solution=None, name="Shifted Schwefel’s Problem 1.2"):
        self.name = name
        shift_data = self.shift_data[:self.dimensions]
        z = solution - shift_data
        return schwefel__(z)
    
    def F20(self, solution=None, name="Shifted Rosenbrock’s Function"):
        self.name = name
        shift_data = self.shift_data[:self.dimensions]
        z = solution - shift_data
        return rosenbrock__(z)
