from sympy import (
    Dummy,
    Expr,
    Function,
    Matrix,
    Symbol,
    cos,
    simplify,
    sin,
    symbols,
    tan,
)
from reporter import MyReporter

def coords(phi, theta):
    return tuple([
        sin(phi) * cos(theta),
        sin(phi) * sin(theta),
        cos(phi)
    ])


def example_pendulum():
    t, m = symbols('t m')
    L = Function('\\ell')(t)
    q = (
        Symbol('\\varphi', real=True),
        Symbol('theta', real=True),
    )
    v = symbols('v:2')

    V: Expr = symbols('V')
    g = Matrix(
        [
            [ m * (L**2), 0],
            [0, m * (L**2) * (sin(q[0]))**2],
        ]
    )

    report = MyReporter(t,q,v,g,V)
    print(report.generate())


def example_particle():
    t, m = symbols('t m')
    G = symbols('G')
    q = (Symbol('x', real=True),)
    v = (Symbol('u', real=True),)

    F = Function('f')(t, q[0])
    Ft = F.diff(t)

    V = simplify(m * (Ft*F.diff(q[0])*q[0].diff(t)+ Ft**2 /2 + G * F))
    g = Matrix([[m*(1 + F.diff(q[0])**2)]])

    report = MyReporter(t,q,v,g,V)
    print(report.generate())


if __name__ == "__main__":
    example_particle()
