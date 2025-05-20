from typing import Tuple
from sympy import ImmutableDenseNDimArray, Matrix, Symbol, simplify

from renderers import my_latex
from riemmanian import (
    christoffel_symbols_get_from_metric,
    curvature_from_christoffel_symbols,
    ricci_from_curvature
)


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
        g_dot = self.g.diff(self.t)
        g_end: Matrix = simplify(g_inv * g_dot)
        Gamma = christoffel_symbols_get_from_metric(self.g, self.q)
        R = curvature_from_christoffel_symbols(Gamma, self.q)
        Ric = ricci_from_curvature(R, self.q)

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
        p_q = self.generate_coords()
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
        return f"""
\\begin{{align*}}
      {p_R[0][0][0][0]} & = {my_latex(R[0, 0, 0, 0])}
    & {p_R[1][0][0][0]} & = {my_latex(R[1, 0, 0, 0])}
\\\\
      {p_R[0][1][0][0]} & = {my_latex(R[0, 1, 0, 0])}
    & {p_R[1][1][0][0]} & = {my_latex(R[1, 1, 0, 0])}
\\\\
      {p_R[0][0][1][0]} & = {my_latex(R[0, 0, 1, 0])}
    & {p_R[1][0][1][0]} & = {my_latex(R[1, 0, 1, 0])}
\\\\
      {p_R[0][1][1][0]} & = {my_latex(R[0, 1, 1, 0])}
    & {p_R[1][1][1][0]} & = {my_latex(R[1, 1, 1, 0])}
\\\\
      {p_R[0][0][0][1]} & = {my_latex(R[0, 0, 0, 1])}
    & {p_R[1][0][0][1]} & = {my_latex(R[1, 0, 0, 1])}
\\\\
      {p_R[0][1][0][1]} & = {my_latex(R[0, 1, 0, 1])}
    & {p_R[1][1][0][1]} & = {my_latex(R[1, 1, 0, 1])}
\\\\
      {p_R[0][0][1][1]} & = {my_latex(R[0, 0, 1, 1])}
    & {p_R[1][0][1][1]} & = {my_latex(R[1, 0, 1, 1])}
\\\\
      {p_R[0][1][1][1]} & = {my_latex(R[0, 1, 1, 1])}
    & {p_R[1][1][1][1]} & = {my_latex(R[1, 1, 1, 1])}
\\end{{align*}}
""".strip()

    def generate_christoffel_symbols_table(
            self,
            Gamma: ImmutableDenseNDimArray
    ) -> str:
        p_Gamma = self.generate_christoffel_symbols()
        return f"""
\\begin{{align*}}
      {p_Gamma[0][0][0]} & = {my_latex(Gamma[0, 0, 0])}
    & {p_Gamma[1][0][0]} & = {my_latex(Gamma[1, 0, 0])}
    & {p_Gamma[0][1][0]} & = {my_latex(Gamma[0, 1, 0])}
    & {p_Gamma[1][1][0]} & = {my_latex(Gamma[1, 1, 0])}
\\\\
      {p_Gamma[0][0][1]} & = {my_latex(Gamma[0, 0, 1])}
    & {p_Gamma[1][0][1]} & = {my_latex(Gamma[1, 0, 1])}
    & {p_Gamma[0][1][1]} & = {my_latex(Gamma[0, 1, 1])}
    & {p_Gamma[1][1][1]} & = {my_latex(Gamma[1, 1, 1])}
\\,,
\\end{{align*}}
""".strip()

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
