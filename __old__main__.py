from sympy import (
    Matrix,
    latex,
    pprint,
    simplify,
    symbols,
    Function,
    diff,
)
from sympy.functions import sin


def Gamma_of(g: Matrix, t, x, y):
    g_inv = g.inv()
    q = (x, y)

    def Gamma(i, j, k):
        result = (
            g_inv[(k, 0)]
            * (
                diff(g[(i, 0)], q[j])
                + diff(g[(j, 0)], q[i])
                - diff(g[(i, j)], q[0])
            )
            / 2
            + g_inv[(k, 1)]
            * (
                diff(g[(i, 1)], q[j])
                + diff(g[(j, 1)], q[i])
                - diff(g[(i, j)], q[1])
            )
            / 2
        )
        return simplify(result)

    return Gamma



def dGamma_of(g: Matrix, t, x, y):
    g_inv = g.inv()
    q = (x, y)

    def dGamma(i, j, k, l):
        result = (
            diff(g_inv[(k, 0)], q[l])
            * (
                diff(g[(i, 0)], q[j])
                + diff(g[(j, 0)], q[i])
                - diff(g[(i, j)], q[0])
            )
            / 2
            + diff(g_inv[(k, 1)], q[l])
            * (
                diff(g[(i, 1)], q[j])
                + diff(g[(j, 1)], q[i])
                - diff(g[(i, j)], q[1])
            )
            / 2
            +
            g_inv[(k, 0)]
            * (
                diff(g[(i, 0)], q[j], q[l])
                + diff(g[(j, 0)], q[i], q[l])
                - diff(g[(i, j)], q[0], q[l])
            )
            / 2
            + g_inv[(k, 1)]
            * (
                diff(g[(i, 1)], q[j], q[l])
                + diff(g[(j, 1)], q[i], q[l])
                - diff(g[(i, j)], q[1], q[l])
            )
            / 2
        )
        return simplify(result)

    return dGamma


def Curvature_of(g: Matrix, t, x, y):
    Gamma = Gamma_of(g, t, x, y)
    dGamma = dGamma_of(g, t, x, y)

    def R(i, j, k, l):
        return simplify(
            dGamma(j,k,l,i) - dGamma(i,k,l,j)
            + Gamma(i, 0, l) * Gamma(j, k, 0)
            + Gamma(i, 1, l) * Gamma(j, k, 1)
            - Gamma(j, 0, l) * Gamma(i, k, 0)
            - Gamma(j, 1, l) * Gamma(i, k, 1)
        )

    return R



class F(Function):

    @classmethod
    def eval(cls, t, x, y):
        """Prevents evaluation into an explicit expression."""
        return None  # Ensures it remains symbolic

    def fdiff(self, argindex):
        """Defines partial derivatives recursively."""
        t, x, y = self.args
        if argindex == 1:
            return self / (2 * t**2)
        elif argindex == 2:  # dF/dx
            return -x * self / t
        elif argindex == 3:  # dF/dy
            return -y * self / t
        else:
            raise ValueError("Invalid differentiation index")

class L(Function):

    @classmethod
    def eval(cls, t):
        """Prevents evaluation into an explicit expression."""
        return None  # Ensures it remains symbolic



def main():
    t, x, y, m = symbols("t x y m")
    l = L("l")

    g = Matrix(
        [
            [ m * l**2, 0],
            [0, m * l**2 * sin(x)**2],
        ]
    )
    Gamma = Gamma_of(g, t, x, y)
    dGamma = dGamma_of(g, t, x, y)
    R = Curvature_of(g, t, x, y)

    document = f"""
\\documentclass{{article}}

\\title{{draft}}
\\author{{Daniel Torres Moral}}
\\date{{January 2025}}

\\begin{{document}}

\\section{{Introduction}}

\\paragraph{{}}
Let's do some calculations.

$$\\Gamma^x_{{xx}} = {latex(Gamma(0,0,0))}$$
$$\\Gamma^y_{{xx}} = {latex(Gamma(0,0,1))}$$
$$\\Gamma^x_{{xy}} = \\Gamma^x_{{yx}} = {latex(Gamma(0,1,0))}$$
$$\\Gamma^y_{{xy}} = \\Gamma^y_{{yx}} = {latex(Gamma(0,1,1))}$$
$$\\Gamma^x_{{yy}} = {latex(Gamma(1,1,0))}$$
$$\\Gamma^y_{{yy}} = {latex(Gamma(1,1,1))}$$
$$\\delta_x\\Gamma^x_{{xx}} = {latex(dGamma(0,0,0,0))}$$
$$\\delta_x\\Gamma^x_{{yx}} = {latex(dGamma(1,0,0,0))}$$
$$\\delta_x\\Gamma^x_{{xy}} = {latex(dGamma(0,1,0,0))}$$
$$\\delta_x\\Gamma^x_{{yy}} = {latex(dGamma(1,1,0,0))}$$
$$\\delta_x\\Gamma^y_{{xx}} = {latex(dGamma(0,0,1,0))}$$
$$\\delta_x\\Gamma^y_{{yx}} = {latex(dGamma(1,0,1,0))}$$
$$\\delta_x\\Gamma^y_{{xy}} = {latex(dGamma(0,1,1,0))}$$
$$\\delta_x\\Gamma^y_{{yy}} = {latex(dGamma(1,1,1,0))}$$
$$\\delta_y\\Gamma^x_{{xx}} = {latex(dGamma(0,0,0,1))}$$
$$\\delta_y\\Gamma^x_{{yx}} = {latex(dGamma(1,0,0,1))}$$
$$\\delta_y\\Gamma^x_{{xy}} = {latex(dGamma(0,1,0,1))}$$
$$\\delta_y\\Gamma^x_{{yy}} = {latex(dGamma(1,1,0,1))}$$
$$\\delta_y\\Gamma^y_{{xx}} = {latex(dGamma(0,0,1,1))}$$
$$\\delta_y\\Gamma^y_{{yx}} = {latex(dGamma(1,0,1,1))}$$
$$\\delta_y\\Gamma^y_{{xy}} = {latex(dGamma(0,1,1,1))}$$
$$\\delta_y\\Gamma^y_{{yy}} = {latex(dGamma(1,1,1,1))}$$
$$R_{{xxx}}^x = {latex(R(0,0,0,0))}$$
$$R_{{yxx}}^x = {latex(R(1,0,0,0))}$$
$$R_{{xyx}}^x = {latex(R(0,1,0,0))}$$
$$R_{{yyx}}^x = {latex(R(1,1,0,0))}$$
$$R_{{xxy}}^x = {latex(R(0,0,1,0))}$$
$$R_{{yxy}}^x = {latex(R(1,0,1,0))}$$
$$R_{{xyy}}^x = {latex(R(0,1,1,0))}$$
$$R_{{yyy}}^x = {latex(R(1,1,1,0))}$$
$$R_{{xxx}}^y = {latex(R(0,0,0,1))}$$
$$R_{{yxx}}^y = {latex(R(1,0,0,1))}$$
$$R_{{xyx}}^y = {latex(R(0,1,0,1))}$$
$$R_{{yyx}}^y = {latex(R(1,1,0,1))}$$
$$R_{{xxy}}^y = {latex(R(0,0,1,1))}$$
$$R_{{yxy}}^y = {latex(R(1,0,1,1))}$$
$$R_{{xyy}}^y = {latex(R(0,1,1,1))}$$
$$R_{{yyy}}^y = {latex(R(1,1,1,1))}$$

\\end{{document}}
""".strip()

    print(f"{document}")

if __name__ == "__main__":
    main()
