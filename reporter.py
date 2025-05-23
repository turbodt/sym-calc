from typing import List, Tuple
from sympy import Dummy, Expr, Function, ImmutableDenseNDimArray, Matrix, MutableDenseNDimArray, Symbol, simplify

from renderers import my_latex
from riemmanian import (
    christoffel_symbols_get_from_metric,
    curvature_from_christoffel_symbols,
    ricci_from_christoffel_symbols
)
def flatten_tuples(t):
    for x in t:
        if isinstance(x, tuple) or isinstance(x, list):
            yield from flatten_tuples(x)
        else:
            yield x

class LatexReporter(object):
    def generate_start(self, title: str) -> str:
        return "\n".join([
            r"\documentclass{article}",
            r"\usepackage{amsmath}",
            r"",
            r"\begin{document}",
            r"",
            r"\section*{%s}" % (title),
            r"",
        ])

    def generate_end(self) -> str:
        return "\n".join([
            r"",
            r"\end{document}",
        ])

class MyReporter(LatexReporter):

    def __init__(
        self,
        t: Symbol,
        q: Tuple[Symbol],
        v: Tuple[Symbol],
        g: Matrix,
        V: Expr
    ):
        self.t = t
        self.q = q
        self.v = v
        self.g = g
        self.g_inv: Matrix = g.inv()
        self.g_dot: Matrix = self.g.diff(self.t)
        self.g_end: Matrix = self.g_inv * self.g_dot
        self.Gamma = christoffel_symbols_get_from_metric(self.g, self.q)
        self.dim = len(q)
        self.V = V
        self.gamma = tuple([Function(f'gamma{i}')(t) for i in range(self.dim)])

        dVdq: List[Expr] = [V.diff(q_i) for q_i in self.q]
        dVddotq: List[Expr] = [V.diff(v_i) for v_i in self.v]
        for k in range(self.dim):
            for i in range(self.dim):
                dVddotq[k] = dVddotq[k].subs(self.q[i], self.gamma[i])
                dVddotq[k] = dVddotq[k].subs(self.v[i], self.gamma[i].diff(t))
        ddVddotqdt: List[Expr] = [expr.diff(t) for expr in dVddotq]
        for k in range(self.dim):
            for i in range(self.dim):
                dVddotq[k] = dVddotq[k].subs(self.gamma[i], self.q[i])
        self.V_euler_lagrange: Tuple[Expr, ...] = tuple([
            dVdq[i] - ddVddotqdt[i] for i in range(self.dim)
        ])

    def generate(self) -> str:

        # do not derive phi nor theta with respect t
        R = curvature_from_christoffel_symbols(self.Gamma, self.q)
        Ric = ricci_from_christoffel_symbols(self.Gamma, self.q)

        result = self.generate_start("Calculations on pendulum")
        result += f"""

Potential energy is:

\\begin{{align*}}
V = {my_latex(self.V)}
\\end{{align*}}

\\bigskip

Given the metric
$$
g = {my_latex(self.g)}
$$
the inverse and derivative matrixes are
\\begin{{align*}}
g^{{-1}} &= {my_latex(self.g_inv)}
&
\\dot g &= {my_latex(self.g_dot)}
\\,.
\\end{{align*}}
In consequence we have
$$
g^{{-1}} \\circ \\dot g
=
{my_latex(self.g_end)}
\\,.
$$
We also obtain the Christoffel Symbols
{self.generate_christoffel_symbols_table(self.Gamma)}

\\bigskip

From here we can deduce the following equations for geodesics
{self.generate_geodesic_equations()}

\\subsection*{{Curvature}}
The curvature tensor is
{self.generate_curvature_table(R)}
and Ricci tensor
$$
\\operatorname{{Ric}} =
{my_latex(Ric)}
\\,.
$$
"""
        result += self.generate_end()
        return result

    def generate_geodesic_equations(self) -> str:
        g_end = self.g_end
        for i in range(self.dim):
            g_end = g_end.subs(self.q[i], self.gamma[i])
            g_end = g_end.subs(self.v[i], self.gamma[i].diff(self.t))

        result = r"\begin{align}"
        t = self.t
        for k in range(self.dim):
            expr = self.gamma[k].diff(t,t)

            for i in range(self.dim):
                for j in range(self.dim):
                    expr += self.Gamma[i,j,k] * self.gamma[i].diff(t) * self.gamma[j].diff(t)

            for i in range(self.dim):
                expr += g_end[k, i] * self.gamma[i].diff(t)

            expr = simplify(expr)

            for i in range(self.dim):
                expr = expr.subs(self.gamma[i], self.q[i])

            if k:
                result += "\\\\\n"
            result += f"{my_latex(self.V_euler_lagrange[k])} &= {my_latex(expr)}"

        result += r"\end{align}"
        return result

    def generate_curvature_table(self, R: ImmutableDenseNDimArray) -> str:
        p_R = self.generate_curvature_symbols()
        symbol_list = tuple(flatten_tuples([
            [
                [
                    [
                        p_R[i][j][k][l]
                        for i in range(self.dim)
                    ]
                    for j in range(self.dim)
                ]
                for k in range(self.dim)
            ]
            for l in range(self.dim)
        ]))
        value_list = tuple(flatten_tuples([
            [
                [
                    [
                        my_latex(R[i, j, k, l])
                        for i in range(self.dim)
                    ]
                    for j in range(self.dim)
                ]
                for k in range(self.dim)
            ]
            for l in range(self.dim)
        ]))

        result = "\\begin{align*}\n"
        for i in range(len(symbol_list)):
            if i %2:
                result += "   & "
            elif i > 0:
                result += "\\\\\n"
            result += f"{symbol_list[i]} & = {value_list[i]}"
        result += r"\end{align*}"
        return result

    def generate_christoffel_symbols_table(
            self,
            Gamma: ImmutableDenseNDimArray
    ) -> str:
        p_Gamma = self.generate_christoffel_symbols()
        symbol_list = tuple(flatten_tuples([
            [
                [
                    p_Gamma[i][j][k]
                    for i in range(self.dim)
                ]
                for j in range(self.dim)
            ]
            for k in range(self.dim)
        ]))
        value_list = tuple(flatten_tuples([
            [
                [
                    my_latex(Gamma[i, j, k])
                    for i in range(self.dim)
                ]
                for j in range(self.dim)
            ]
            for k in range(self.dim)
        ]))

        result = "\\begin{align*}\n"
        for i in range(len(symbol_list)):
            if i %4:
                result += "   & "
            elif i > 0:
                result += "\\\\\n"
            result += f"{symbol_list[i]} & = {value_list[i]}"
        result += r"\end{align*}"
        return result

    def generate_ricci_symbols(self) -> Tuple[Tuple[str,...],...]:
        p_q = self.generate_coords()
        return tuple([
            tuple([
                f"\\operatorname{{Ric}}_{{{p_q[i]}{p_q[j]}}}"
                for j in range(self.dim)
            ])
            for i in range(self.dim)
        ])

    def generate_curvature_symbols(self) -> Tuple[Tuple[Tuple[Tuple[str,...],...],...],...]:
        p_q = self.generate_coords()
        return tuple([
            tuple([
                tuple([
                    tuple([
                        f"R_{{{p_q[i]}{p_q[j]}{p_q[k]}}}^{{{p_q[l]}}}"
                        for l in range(self.dim)
                    ])
                    for k in range(self.dim)
                ])
                for j in range(self.dim)
            ])
            for i in range(self.dim)
        ])

    def generate_christoffel_symbols(self) -> Tuple[Tuple[Tuple[str,...],...],...]:
        p_q = self.generate_coords()
        return tuple([
            tuple([
                tuple([
                    f"\\Gamma_{{{p_q[i]}{p_q[j]}}}^{{{p_q[k]}}}"
                    for k in range(self.dim)
                ])
                for j in range(self.dim)
            ])
            for i in range(self.dim)
        ])

    def generate_coords(self) -> Tuple[str,...]:
        return tuple([my_latex(q_i) for q_i in self.q])
