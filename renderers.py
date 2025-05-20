from sympy import symbols
from sympy.printing.latex import LatexPrinter


class DotNotationPrinter(LatexPrinter):
    def _print_Derivative(self, expr):
        base = self._print(expr.expr)
        var = expr.variables

        if all(v == symbols('t') for v in var):
            dots = len(var)
            return r'\%sot{%s}' % ('d' * dots, base)
        else:
            return super()._print_Derivative(expr)


def my_latex(expr):
    return DotNotationPrinter().doprint(expr)
