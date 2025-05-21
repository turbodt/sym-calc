from sympy import (
    Dummy,
    Expr,
    Function,
    Matrix,
    cos,
    simplify,
    sin,
    symbols,
)
from reporter import MyReporter

def coords(phi, theta):
    return tuple([
        sin(phi) * cos(theta),
        sin(phi) * sin(theta),
        cos(phi)
    ])


def custom_display(val: str):
    def display(self, *args, **kwargs):
        return val
    return display


if __name__ == "__main__":
    t, m = symbols('t m')
    q = (Function('phi', real=True)(t), Function('theta', real=True)(t), )
    q[0]._latex = custom_display(r"\varphi")
    q[1]._latex = custom_display(r"\theta")
    n = len(q)
    L = Function('l')(t)
    G = symbols('G')
    L._latex = custom_display(r"\ell")

    g = Matrix(
        [
            [ m * (L**2), 0],
            [0, m * (L**2) * (sin(q[0]))**2],
        ]
    )
    V: Expr = symbols('V')
    report = MyReporter(t,q,g,V)
    print(report.generate())

    x = (Function('x', real=True)(t),)
    x[0]._latex = custom_display(r"x")
    F = Function('f')(t, x[0])
    #F._latex = custom_display(r"f")
    u = [Dummy(f"u{i}",real=True) for i in range(len(x))]
    Ft = F.subs({x[i]: u[i] for i in range(len(x))})
    Ft = Ft.diff(t)
    Ft = Ft.subs({u[i]: x[i] for i in range(len(x))})

    g2 = Matrix([[m*(1 + F.diff(x[0])**2)]])

    V = simplify(m * (Ft*F.diff(x[0])*x[0].diff(t)+ Ft**2 /2 + G * F))


    report = MyReporter(t,x,g2,V)
    #print(report.generate())
