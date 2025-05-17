from math import exp
from typing import Tuple
from sympy import Function, Matrix, Symbol, cos, latex, pprint, pretty, sin, symbols
from riemmanian import christoffel_symbols_get_from_metric, curvature_from_christoffel_symbols


class TimeDependentFunction(Function):
    @classmethod
    def eval(cls, t: Symbol):
        return None


def e(t):
    return exp(t)


def f(t, *x):
    n = len(x) - 1
    a = e(t)**2 * (1 + x[n]) - (1 - x[n])
    b = e(t)**2 * (1 + x[n]) + (1 - x[n])
    return tuple([ 2*e(t)*x[i]/b for i in range(n)] + [a])


def coords(phi, theta):
    return tuple([
        sin(phi) * cos(theta),
        sin(phi) * sin(theta),
        cos(phi)
    ])


if __name__ == "__main__":
    t, m = symbols('t m')
    q = symbols('phi theta')
    n = len(q)
    L: TimeDependentFunction = TimeDependentFunction('L')

    g = Matrix(
        [
            [ m * (L**2), 0],
            [0, m * (L**2) * (sin(q[0]))**2],
        ]
    )


    Gamma = christoffel_symbols_get_from_metric(g, q)
    R = curvature_from_christoffel_symbols(Gamma, q)

    #print(f"{latex(R)}")
