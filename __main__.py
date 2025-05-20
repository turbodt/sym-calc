from sympy import (
    Function,
    Matrix,
    cos,
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
    L._latex = custom_display(r"\ell")

    g = Matrix(
        [
            [ m * (L**2), 0],
            [0, m * (L**2) * (sin(q[0]))**2],
        ]
    )

    report = MyReporter(t,q,g)
    print(report.generate())
