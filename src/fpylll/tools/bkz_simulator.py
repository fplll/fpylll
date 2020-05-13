# -*- coding: utf-8 -*-
"""
BKZ simulation algorithm as proposed in

- Chen, Y., & Nguyen, P. Q. (2011). BKZ 2.0: better lattice security estimates. In D. H. Lee, & X.
  Wang, ASIACRYPT~2011 (pp. 1–20). : Springer, Heidelberg.

.. moduleauthor:: Michael Walter <fplll-devel@googlegroups.com> (2014)
.. moduleauthor:: Martin R. Albrecht <fplll-devel@googlegroups.com> (2018)

"""
from copy import copy
from math import log, sqrt, lgamma, pi
from collections import OrderedDict

from fpylll.tools.quality import basis_quality
from fpylll.tools.bkz_stats import pretty_dict
from fpylll.fplll.bkz import BKZ
from fpylll.fplll.integer_matrix import IntegerMatrix
from fpylll.fplll.gso import MatGSO, GSO

rk = (
    0.789527997160000,
    0.780003183804613,
    0.750872218594458,
    0.706520454592593,
    0.696345241018901,
    0.660533841808400,
    0.626274718790505,
    0.581480717333169,
    0.553171463433503,
    0.520811087419712,
    0.487994338534253,
    0.459541470573431,
    0.414638319529319,
    0.392811729940846,
    0.339090376264829,
    0.306561491936042,
    0.276041187709516,
    0.236698863270441,
    0.196186341673080,
    0.161214212092249,
    0.110895134828114,
    0.0678261623920553,
    0.0272807162335610,
    -0.0234609979600137,
    -0.0320527224746912,
    -0.0940331032784437,
    -0.129109087817554,
    -0.176965384290173,
    -0.209405754915959,
    -0.265867993276493,
    -0.299031324494802,
    -0.349338597048432,
    -0.380428160303508,
    -0.427399405474537,
    -0.474944677694975,
    -0.530140672818150,
    -0.561625221138784,
    -0.612008793872032,
    -0.669011014635905,
    -0.713766731570930,
    -0.754041787011810,
    -0.808609696192079,
    -0.859933249032210,
    -0.884479963601658,
    -0.886666930030433,
)


def simulate(r, param):
    """
    BKZ simulation algorithm as proposed by Chen and Nguyen in "BKZ 2.0: Better Lattice Security
    Estimates".  Returns the reduced squared norms of the GSO vectors of the basis and the number of
    BKZ tours simulated.  This version terminates when no substantial progress is made anymore or at
    most ``max_loops`` tours were simulated.  If no ``max_loops`` is given, at most ``d`` tours are
    performed, where ``d`` is the dimension of the lattice.

    :param r: squared norms of the GSO vectors of the basis.
    :param param: BKZ parameters

    EXAMPLE:

        >>> from fpylll import IntegerMatrix, GSO, LLL, FPLLL, BKZ
        >>> FPLLL.set_random_seed(1337)
        >>> A = LLL.reduction(IntegerMatrix.random(100, "qary", bits=30, k=50))
        >>> M = GSO.Mat(A)

        >>> from fpylll.tools.bkz_simulator import simulate
        >>> _ = simulate(M, BKZ.Param(block_size=40, max_loops=4, flags=BKZ.VERBOSE))
        {"i":        0,  "r_0":   2^33.3,  "r_0/gh": 6.110565,  "rhf": 1.018340,  "/": -0.07013,  "hv/hv": 2.424131}
        {"i":        1,  "r_0":   2^32.7,  "r_0/gh": 4.018330,  "rhf": 1.016208,  "/": -0.06161,  "hv/hv": 2.156298}
        {"i":        2,  "r_0":   2^32.3,  "r_0/gh": 2.973172,  "rhf": 1.014679,  "/": -0.05745,  "hv/hv": 2.047014}
        {"i":        3,  "r_0":   2^32.1,  "r_0/gh": 2.583479,  "rhf": 1.013966,  "/": -0.05560,  "hv/hv": 2.000296}


    """

    if isinstance(r, IntegerMatrix):
        r = GSO.Mat(r)
    if isinstance(r, MatGSO):
        r.update_gso()
        r = r.r()

    d = len(r)

    # code uses log2 of norms, FPLLL uses squared norms
    r = list(map(lambda x: log(x, 2) / 2.0, r))

    r1 = copy(r)
    r2 = copy(r)
    c = [rk[-i] - sum(rk[-i:]) / i for i in range(1, 46)]
    c += [
        (lgamma(beta / 2.0 + 1) * (1.0 / beta) - log(sqrt(pi))) / log(2.0)
        for beta in range(46, param.block_size + 1)
    ]

    if param.max_loops:
        max_loops = param.max_loops
    else:
        max_loops = d

    for i in range(max_loops):
        phi = True
        for k in range(d - min(45, param.block_size)):
            beta = min(param.block_size, d - k)
            f = k + beta
            logV = sum(r1[:f]) - sum(r2[:k])
            lma = logV / beta + c[beta - 1]
            if phi:
                if lma < r1[k]:
                    r2[k] = lma
                    phi = False
            else:
                r2[k] = lma

        # early termination
        if phi or r1 == r2:
            break
        else:
            beta = min(45, param.block_size)
            logV = sum(r1) - sum(r2[:-beta])

            if param.block_size < 45:
                tmp = sum(rk[-param.block_size :]) / param.block_size
                rk1 = [r_ - tmp for r_ in rk[-param.block_size :]]
            else:
                rk1 = rk

            for k, r in zip(range(d - beta, d), rk1):
                r2[k] = logV / beta + r
            r1 = copy(r2)

        if param.flags & BKZ.VERBOSE:
            r = OrderedDict()
            r["i"] = i
            for k, v in basis_quality(list(map(lambda x: 2.0 ** (2 * x), r1))).items():
                r[k] = v
            print(pretty_dict(r))

    r1 = list(map(lambda x: 2.0 ** (2 * x), r1))
    return r1, i + 1
