from typing import Tuple
from sympy import (
    Dummy,
    Expr,
    ImmutableDenseNDimArray,
    Matrix,
    MutableDenseNDimArray,
    Symbol,
    diff,
    simplify,
)


def dummy_simplification(expr: Expr, q: Tuple[Symbol]):
    u = [Dummy(f'u{i}', real=True) for i in range(len(q))]
    temp = expr.subs({q[i]: u[i] for i in range(len(q))})
    temp = simplify(temp)
    return temp.subs({u[i]: q[i] for i in range(len(q))})


def christoffel_symbols_get_from_metric(
    g: Matrix,
    q: Tuple[Symbol]
) -> ImmutableDenseNDimArray:
    g_inv = g.inv()
    n = len(q)

    Gamma = [[[0 for _ in range(n)] for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            for k in range(n):
                expr = 0
                for l in range(n):
                    expr += g_inv[k, l] * (
                        diff(g[l, j], q[i]) +
                        diff(g[l, i], q[j]) -
                        diff(g[j, i], q[l])
                    )
                Gamma[i][j][k] = dummy_simplification(expr / 2, q)
    return ImmutableDenseNDimArray(Gamma)


def curvature_from_christoffel_symbols(
    Gamma: ImmutableDenseNDimArray,
    q: Tuple[Symbol]
) -> ImmutableDenseNDimArray:

    n = len(q)
    R = MutableDenseNDimArray.zeros(n,n,n,n)

    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    expr: Expr = (
                        diff(Gamma[k][i][l], q[j])
                        - diff(Gamma[j][i][l], q[k])
                    )
                    for h in range(n):
                        expr += (
                            + Gamma[j][h][l] * Gamma[k][i][h]
                            - Gamma[k][h][l] * Gamma[j][i][h]
                        )
                    R[i, j, k, l] = dummy_simplification(expr, q)
    return ImmutableDenseNDimArray(R)


def ricci_from_christoffel_symbols(
    Gamma: ImmutableDenseNDimArray,
    q: Tuple[Symbol]
) -> ImmutableDenseNDimArray:

    n = len(q)
    Ric = MutableDenseNDimArray.zeros(n,n)

    for i in range(n):
        for k in range(n):
            expr: Expr = 0
            for j in range(n):
                expr += diff(Gamma[k][i][j], q[j]) - diff(Gamma[j][i][j], q[k])
                for l in range(n):
                    expr += (
                        + Gamma[j][l][j] * Gamma[k][i][l]
                        - Gamma[k][l][j] * Gamma[j][i][l]
                    )
            Ric[i, k] = dummy_simplification(expr, q)
    return ImmutableDenseNDimArray(Ric)

