from typing import Tuple
from sympy import Dummy, ImmutableDenseNDimArray, Matrix, Symbol, simplify

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

    def __init__(self, t: Symbol, q: Tuple[Symbol], g: Matrix):
        self.t = t
        self.q = q
        self.g = g
        self.dim = len(q)

    def generate(self) -> str:

        g_inv = self.g.inv()

        # do not derive phi nor theta with respect t
        u = [Dummy(f"u{i}",real=True) for i in range(self.dim)]
        g_temp = self.g.subs({self.q[i]: u[i] for i in range(self.dim)})
        g_dot = g_temp.diff(self.t)
        g_dot = g_dot.subs({u[i]: self.q[i] for i in range(self.dim)})

        g_end: Matrix = simplify(g_inv * g_dot)
        Gamma = christoffel_symbols_get_from_metric(self.g, self.q)
        R = curvature_from_christoffel_symbols(Gamma, self.q)
        Ric = ricci_from_christoffel_symbols(Gamma, self.q)

        result = self.generate_start("Calculations on pendulum")
        result += f"""
Given the metric
$$
g = {my_latex(self.g)}
$$
the inverse and derivative matrixes are
\\begin{{align*}}
g^{{-1}} &= {my_latex(g_inv)}
&
\\dot g &= {my_latex(g_dot)}
\\,.
\\end{{align*}}
In consequence we have
$$
g^{{-1}} \\circ \\dot g
=
{my_latex(g_end)}
\\,.
$$
We also obtain the Christoffel Symbols
{self.generate_christoffel_symbols_table(Gamma)}

\\bigskip

From here we can deduce the following equations for geodesics
{self.generate_geodesic_equations(Gamma, g_end)}

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

    def generate_geodesic_equations(
        self,
        Gamma: ImmutableDenseNDimArray,
        g_end: Matrix
    ) -> str:
        result = r"\begin{align}"
        t = self.t
        for k in range(self.dim):
            expr = self.q[k].diff(t, t)

            for i in range(self.dim):
                for j in range(self.dim):
                    expr += Gamma[i,j,k] * self.q[i].diff(t) * self.q[j].diff(t)

            for i in range(self.dim):
                expr += g_end[k, i] * self.q[i].diff(t)

            expr = simplify(expr)

            if k:
                result += "\\\\\n"
            result += f"0 &= {my_latex(expr)}"

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
