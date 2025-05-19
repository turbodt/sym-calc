from typing import Tuple
from sympy import (
    DenseNDimArray,
    Expr,
    ImmutableDenseNDimArray,
    Matrix,
    MutableDenseNDimArray,
    Symbol,
    diff,
    simplify,
)


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
                    expr += g_inv[i, l] * (
                        diff(g[l, j], q[k]) +
                        diff(g[l, k], q[j]) -
                        diff(g[j, k], q[l])
                    )
                Gamma[i][j][k] = simplify(0.5 * expr)
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
                    R[i, j, k, l] = simplify(expr)
    return ImmutableDenseNDimArray(R)


def ricci_from_curvature(
    R: ImmutableDenseNDimArray,
    q: Tuple[Symbol]
) -> ImmutableDenseNDimArray:
    n = len(q)

    Ric = MutableDenseNDimArray.zeros(n,n)

    for i in range(n):
        for j in range(n):
            for k in range(n):
                Ric[i, k] += R[i, j, k, j]

    return ImmutableDenseNDimArray(Ric)
