# -*- coding: utf-8 -*-
"""
This file implements two BKZ simulation algorithms as proposed in

- Chen, Y., & Nguyen, P. Q. (2011). BKZ 2.0: better lattice security estimates. In D. H. Lee, & X.
  Wang, ASIACRYPT~2011 (pp. 1–20). : Springer, Heidelberg.

- Bai, S., & Stehlé, D. & Wen, W. (2018). Measuring, simulating and exploiting
  the head concavity phenomenon in BKZ. In T. Peyrin, & S. Galbraith,
  ASIACRYPT~2018 : Springer, Heidelberg.

.. moduleauthor:: Michael Walter <fplll-devel@googlegroups.com> (2014)
.. moduleauthor:: Martin R. Albrecht <fplll-devel@googlegroups.com> (2018)
.. moduleauthor:: Shi Bai <fplll-devel@googlegroups.com> (2020)
.. moduleauthor:: Fernando Virdia <fernando.virdia.2016@rhul.ac.uk> (2020)

"""
from copy import copy
from math import log, sqrt, lgamma, pi, exp
from collections import OrderedDict
import random

from fpylll.tools.quality import basis_quality
from fpylll.tools.bkz_stats import pretty_dict
from fpylll.fplll.bkz import BKZ
from fpylll.fplll.integer_matrix import IntegerMatrix
from fpylll.fplll.gso import MatGSO, GSO
from fpylll import FPLLL

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


def _extract_log_norms(r):
    if isinstance(r, IntegerMatrix):
        r = GSO.Mat(r)
    elif isinstance(r, MatGSO):
        r.update_gso()
        r = r.r()
    else:
        for ri in r:
            if (ri <= 0):
                raise ValueError("squared norms in r should be positive")

    # code uses log2 of norms, FPLLL uses squared norms
    r = list(map(lambda x: log(x, 2) / 2.0, r))
    return r


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

    r = _extract_log_norms(r)

    d = len(r)

    r1 = copy(r)
    r2 = copy(r)
    c = [rk[-i] - sum(rk[-i:]) / i for i in range(1, 46)]
    c += [
        (lgamma(beta / 2.0 + 1) * (1.0 / beta) - log(sqrt(pi))) / log(2.0)
        for beta in range(46, param.block_size + 1)
    ]

    if param.max_loops:
        N = param.max_loops
    else:
        N = d

    for i in range(N):
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


def simulate_prob(r, param, prng_seed=0xdeadbeef):
    """
    BKZ simulation algorithm as proposed by Bai and Stehlé and Wen in "Measuring, simulating and
    exploiting the head concavity phenomenon in BKZ".  Returns the reduced squared norms of the
    GSO vectors of the basis and the number of BKZ tours simulated.  This version terminates when
    no substantial progress is made anymore or at most ``max_loops`` tours were simulated.
    If no ``max_loops`` is given, at most ``d`` tours are performed, where ``d`` is the dimension
    of the lattice.
    :param r: squared norms of the GSO vectors of the basis.
    :param param: BKZ parameters
    EXAMPLE:
        >>> from fpylll import IntegerMatrix, GSO, LLL, FPLLL, BKZ
        >>> FPLLL.set_random_seed(1337)
        >>> A = LLL.reduction(IntegerMatrix.random(100, "qary", bits=30, k=50))
        >>> M = GSO.Mat(A)
        >>> from fpylll.tools.bkz_simulator import simulate_prob
        >>> _ = simulate_prob(M, BKZ.Param(block_size=40, max_loops=4, flags=BKZ.VERBOSE))
        {"i":        0,  "r_0":   2^33.1,  "r_0/gh": 5.193166,  "rhf": 1.017512,  "/": -0.07022,  "hv/hv": 2.428125}
        {"i":        1,  "r_0":   2^32.7,  "r_0/gh": 3.997766,  "rhf": 1.016182,  "/": -0.06214,  "hv/hv": 2.168460}
        {"i":        2,  "r_0":   2^32.3,  "r_0/gh": 3.020156,  "rhf": 1.014759,  "/": -0.05808,  "hv/hv": 2.059562}
        {"i":        3,  "r_0":   2^32.2,  "r_0/gh": 2.783102,  "rhf": 1.014344,  "/": -0.05603,  "hv/hv": 2.013191}
    """

    if param.block_size <= 2:
        raise ValueError("The BSW18 simulator requires block size >= 3.")

    # fix PRNG seed
    random.seed(prng_seed if prng_seed else FPLLL.randint(0, 2**32-1))

    r = _extract_log_norms(r)

    d = len(r)

    r1 = copy(r)
    r2 = copy(r)
    c = [rk[-j] - sum(rk[-j:]) / j for j in range(1, 46)]
    c += [
        (lgamma(beta / 2.0 + 1) * (1.0 / beta) - log(sqrt(pi))) / log(2.0)
        for beta in range(46, param.block_size + 1)
    ]

    if param.max_loops:
        N = param.max_loops
    else:
        N = d

    t0 = [True for _ in range(d)]
    for i in range(N):
        t1 = [False for _ in range(d)]
        for k in range(d - min(45, param.block_size)):
            beta = min(param.block_size, d - k)
            f = k + beta
            phi = False
            for kp in range(k, f):
                phi |= t0[kp]
            logV = sum(r1[:f]) - sum(r2[:k])
            if phi:
                X = random.expovariate(.5)
                lma = (log(X, 2) + logV) / beta + c[beta - 1]
                if lma < r1[k]:
                    r2[k] = lma
                    r2[k+1] = r1[k] + log(sqrt(1-1./beta), 2)
                    dec = (r1[k]-lma) + (r1[k+1] - r2[k+1])
                    for j in range(k+2, f):
                        r2[j] = r1[j] + dec/(beta-2.)
                        t1[j] = True
                    phi = False

            for j in range(k, f):
                r1[j] = r2[j]

        # early termination
        if True not in t1:
            break

        # last block
        beta = min(45, param.block_size)
        logV = sum(r1) - sum(r2[:-beta])
        if param.block_size < 45:
            rk1 = normalize_GSO_unitary(rk[-beta:])
        else:
            rk1 = rk
        K = range(d-beta, d)
        for k, r in zip(K, rk1):
            r2[k] = logV / beta + r
            t1[k] = True

        # early termination
        if (r1 == r2):
            break
        r1 = copy(r2)
        t0 = copy(t1)

        if param.flags & BKZ.VERBOSE:
            r = OrderedDict()
            r["i"] = i
            for k, v in basis_quality(list(map(lambda x: 2.0 ** (2 * x), r1))).items():
                r[k] = v
            print(pretty_dict(r))

    r1 = list(map(lambda x: 2.0 ** (2 * x), r1))
    return r1, i + 1


def normalize_GSO_unitary(l):
    log_det = sum(l)
    n = len(l)
    nor_log_det = [0.0] * n
    for i in range(n):
        nor_log_det[i] = l[i] - log_det/n
    return nor_log_det


def averaged_simulate_prob(L, param, tries=10):
    """
    This wrapper calls the [BSW18] probabilistic BKZ simulator with different
    PRNG seeds, and returns the average output.
    :param r: squared norms of the GSO vectors of the basis.
    :param param: BKZ parameters
    :tries: number of iterations to average over. Default: 10
    EXAMPLE:
        >>> from fpylll import IntegerMatrix, GSO, LLL, FPLLL, BKZ
        >>> FPLLL.set_random_seed(1337)
        >>> A = LLL.reduction(IntegerMatrix.random(100, "qary", bits=30, k=50))
        >>> M = GSO.Mat(A)
        >>> from fpylll.tools.bkz_simulator import averaged_simulate_prob
        >>> _ = averaged_simulate_prob(M, BKZ.Param(block_size=40, max_loops=4))
        >>> print(_[0][:3])
        [4663149828.487..., 4267813469.1884..., 4273411937.5775...]
    """
    if tries < 1:
        raise ValueError("Need to average over positive number of tries.")

    for _ in range(tries):
        x, y = simulate_prob(L, param, prng_seed=_+1)
        x = list(map(log, x))
        if _ == 0:
            i = [l for l in x]
            j = y
        else:
            inew = [sum(l) for l in zip(i, x)]
            i = inew
            j += y

    i = [l/tries for l in i]
    j = j/tries
    return list(map(exp, i)), j
