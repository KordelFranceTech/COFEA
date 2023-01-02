import numba
from numpy import cos, sqrt, pi, e, exp, sum


@numba.jit
def sphere__(solution=None):
    return sum(solution ** 2)


@numba.jit
def elliptic__(solution=None):
    result = 0.0
    for i in range(0, len(solution)):
        result += (10 ** 6) ** (i / (len(solution) - 1)) * solution[i] ** 2
    return result


@numba.jit
def rastrigin__(solution=None):
    return sum(solution ** 2 - 10 * cos(2 * pi * solution) + 10)


@numba.jit
def ackley__(solution=None):
    return -20 * exp(-0.2 * sqrt(sum(solution ** 2) / len(solution))) - exp(
        sum(cos(2 * pi * solution)) / len(solution)) + 20 + e


@numba.jit
def schwefel__(solution=None):
    return sum([sum(solution[:i]) ** 2 for i in range(0, len(solution))])


@numba.jit
def rosenbrock__(solution=None):
    result = 0.0
    for i in range(len(solution) - 1):
        result += 100 * (solution[i] ** 2 - solution[i + 1]) ** 2 + (solution[i] - 1) ** 2
    return result
