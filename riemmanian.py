from typing import Tuple
from sympy import (
    DenseNDimArray,
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
                    expr = 0
                    for h in range(n):
                        expr += (
                            diff(Gamma[j][k][h], q[i])
                            - diff(Gamma[i][k][h], q[j])
                            + Gamma[i][l][h] * Gamma[j][k][l]
                            - Gamma[j][l][h] * Gamma[i][k][l]
                        )
                    R[i, j, k, l] = simplify(expr)
    return ImmutableDenseNDimArray(R)
