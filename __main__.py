from sympy import (
    Function,
    Matrix,
    Symbol,
    cos,
    latex,
    sin,
    symbols,
)
from riemmanian import (
    christoffel_symbols_get_from_metric,
    curvature_from_christoffel_symbols,
    ricci_from_curvature,
)


class TimeDependentFunction(Function):
    @classmethod
    def eval(cls, t: Symbol):
        return None

    def _latex(self, printer, *args, **kwargs):
        pargs = [printer.doprint(arg) for arg in self.args]
        if len(pargs) == 1:
            return r'\operatorname{%s}' % (pargs[0])

        return r'\operatorname{%s}(%s)' % (pargs[0], pargs[1])


class Length(TimeDependentFunction):
    def _latex(self, printer, *args, **kwargs):
        pargs = [printer.doprint(arg) for arg in self.args]
        if len(pargs) == 1:
            return r'\ell'

        return r'\ell(%s)' % (pargs[1])

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
    L: TimeDependentFunction = Length('L')

    g = Matrix(
        [
            [ m * (L**2), 0],
            [0, m * (L**2) * (sin(q[0]))**2],
        ]
    )

    Gamma = christoffel_symbols_get_from_metric(g, q)
    R = curvature_from_christoffel_symbols(Gamma, q)
    Ric = ricci_from_curvature(R, q)

    p_q = [latex(q_i) for q_i in q]
    def p_Gamma(i, j, k):
        return f'\\Gamma_{{{p_q[i]}{p_q[j]}}}^{{{p_q[k]}}}'
    def p_R(i, j, k, l):
        return f'R_{{{p_q[i]}{p_q[j]}{p_q[k]}}}^{{{p_q[l]}}}'
    def p_Ric(i, j):
        return f'\\operatorname{{Ricc}}_{{{p_q[i]}{p_q[j]}}}'

    print(f"""
Given the metric
$$
g = {latex(g)}
$$
we obtain Christoffel Symbols
\\begin{{align*}}
      {p_Gamma(0,0,0)} & = {latex(Gamma[0, 0, 0])}
    & {p_Gamma(1,0,0)} & = {latex(Gamma[1, 0, 0])}
    & {p_Gamma(0,1,0)} & = {latex(Gamma[0, 1, 0])}
    & {p_Gamma(1,1,0)} & = {latex(Gamma[1, 1, 0])}
\\\\
      {p_Gamma(0,0,1)} & = {latex(Gamma[0, 0, 1])}
    & {p_Gamma(1,0,1)} & = {latex(Gamma[1, 0, 1])}
    & {p_Gamma(0,1,1)} & = {latex(Gamma[0, 1, 1])}
    & {p_Gamma(1,1,1)} & = {latex(Gamma[1, 1, 1])}
\\,,
\\end{{align*}}
curature tensor
\\begin{{align*}}
      {p_R(0,0,0,0)} & = {latex(R[0, 0, 0, 0])}
    & {p_R(1,0,0,0)} & = {latex(R[1, 0, 0, 0])}
\\\\
      {p_R(0,1,0,0)} & = {latex(R[0, 1, 0, 0])}
    & {p_R(1,1,0,0)} & = {latex(R[1, 1, 0, 0])}
\\\\
      {p_R(0,0,1,0)} & = {latex(R[0, 0, 1, 0])}
    & {p_R(1,0,1,0)} & = {latex(R[1, 0, 1, 0])}
\\\\
      {p_R(0,1,1,0)} & = {latex(R[0, 1, 1, 0])}
    & {p_R(1,1,1,0)} & = {latex(R[1, 1, 1, 0])}
\\\\
      {p_R(0,0,0,1)} & = {latex(R[0, 0, 0, 1])}
    & {p_R(1,0,0,1)} & = {latex(R[1, 0, 0, 1])}
\\\\
      {p_R(0,1,0,1)} & = {latex(R[0, 1, 0, 1])}
    & {p_R(1,1,0,1)} & = {latex(R[1, 1, 0, 1])}
\\\\
      {p_R(0,0,1,1)} & = {latex(R[0, 0, 1, 1])}
    & {p_R(1,0,1,1)} & = {latex(R[1, 0, 1, 1])}
\\\\
      {p_R(0,1,1,1)} & = {latex(R[0, 1, 1, 1])}
    & {p_R(1,1,1,1)} & = {latex(R[1, 1, 1, 1])}
\\end{{align*}}
and Ricci tensor
$$
\\operatorname{{Ric}} = {latex(Ric)}
\\,.
$$
    """)

