# -*- coding: utf-8 -*-
"""
This file implements two BKZ simulation algorithms as proposed in

- Chen, Y., & Nguyen, P. Q. (2011). BKZ 2.0: better lattice security estimates.
  In D. H. Lee, & X.  Wang, ASIACRYPT~2011 (pp. 1–20). : Springer, Heidelberg.

- Bai, S., & Stehlé, D. & Wen, W. (2018). Measuring, simulating and exploiting
  the head concavity phenomenon in BKZ. In T. Peyrin, & S. Galbraith,
  ASIACRYPT~2018 : Springer, Heidelberg.

.. moduleauthor:: Michael Walter <fplll-devel@googlegroups.com> (2014)
.. moduleauthor:: Martin R. Albrecht <fplll-devel@googlegroups.com> (2018)
.. moduleauthor:: Shi Bai <fplll-devel@googlegroups.com> (2020)
.. moduleauthor:: Fernando Virdia <fernando.virdia.2016@rhul.ac.uk> (2020)
.. moduleauthor:: Ludo Pulles <lnp@cwi.nl> (2024)

"""
from copy import copy
from math import log, sqrt, lgamma, pi
from collections import OrderedDict
import random

from fpylll.tools.quality import basis_quality
from fpylll.tools.bkz_stats import pretty_dict
from fpylll.fplll.bkz import BKZ
from fpylll.fplll.integer_matrix import IntegerMatrix
from fpylll.fplll.gso import MatGSO, GSO
from fpylll import FPLLL

# Line 2, Algorithm 2 of [CN11]:
# Average log2(||b_k*||) of an HKZ-reduced random unit-volume 45-dim lattice.
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


def _lg_gh(d):
    """
    Return the logarithm (base 2) of the Gaussian Heuristic, GH(d).
    Here GH(d) is the expected length of the shortest nonzero vector in a
    random lattice of dimension d.
    :param d: dimension
    """
    if d <= 45:
        return rk[-d] - sum(rk[-d:]) / d
    log_gh = lgamma(d / 2.0 + 1) * (1.0 / d) - log(sqrt(pi))
    return log_gh / log(2.0)


def log_simulate(basis_profile, beta, max_loops=False, verbose=False):
    """
    Simulate the evolution of the basis profile when running multiple tours of
    BKZ-beta, using the simulator Algorithm 1 in [CN11].

    :param basis_profile: log2(norm) of the GSO vectors of the basis.
    :param max_loops: maximum number of full tours in BKZ to perform.
                      BKZ terminates earlier if no progress is made in a tour.
    :param beta: block size for BKZ.
    :returns: tuple containing:
              1. the basis profile after BKZ finished,
              2. number of tours performed.
    """
    profile, n = copy(basis_profile), len(basis_profile)
    assert 2 <= beta <= n

    lg_ghs = [0] + [_lg_gh(i) for i in range(1, beta + 1)]
    if not max_loops:
        max_loops = n

    for j in range(max_loops):
        lg_volume = sum(profile[:beta])  # lg_volume = sum(profile[k:f])
        updated = False
        for k in range(0, n - 1):
            f = min(k + beta, n)  # end index (excl.) of local block [k, f)
            d = f - k  # dimension of local block
            svp_sol = lg_volume/d + lg_ghs[d]
            if updated or svp_sol < profile[k]:
                profile[k] = svp_sol
                updated = True

            lg_volume -= profile[k]  # Remove ||b_k*|| from the sliding window
            if f < n:
                lg_volume += profile[f]  # Add ||b_f*|| to the sliding window
        profile[-1] = lg_volume  # sum(profile) = lg(det L)
        if not updated:
            return profile, j + 1  # Early termination (unlikely)
        if verbose:
            stats = {'i': j} | basis_quality([2.0**(2 * x) for x in profile])
            print(pretty_dict(OrderedDict(stats)))

    return profile, max_loops


def log_simulate_prob(basis_profile, block_size, prng_seed=0xdeadbeef,
                      max_loops=False, verbose=False):
    """
    Simulate the evolution of the basis profile when running multiple tours of
    BKZ-beta, using the simulator by Bai, Stehlé and Wen [BSW18].

    :param basis_profile: log2(norm) of the GSO vectors of the basis.
    :param max_loops: maximum number of full tours in BKZ to perform.
                      BKZ terminates earlier if no progress is made in a tour.
    :param beta: block size for BKZ.
    :returns: tuple containing:
              1. the basis profile after BKZ finished,
              2. number of tours performed.
    """
    if block_size <= 2:
        raise ValueError("The BSW18 simulator requires block size >= 3.")

    # fix PRNG seed
    random.seed(prng_seed if prng_seed else FPLLL.randint(0, 2**32 - 1))

    d = len(basis_profile)

    r1 = copy(basis_profile)
    r2 = copy(basis_profile)
    lg_ghs = [0] + [_lg_gh(i) for i in range(1, block_size + 1)]

    if not max_loops:
        max_loops = d

    t0 = [True for _ in range(d)]
    for i in range(max_loops):
        t1 = [False for _ in range(d)]
        for k in range(d - min(45, block_size)):
            beta = min(block_size, d - k)
            f = k + beta
            phi = False
            for kp in range(k, f):
                phi |= t0[kp]
            lg_volume = sum(r1[:f]) - sum(r2[:k])
            if phi:
                x = random.expovariate(.5)
                lma = (log(x, 2) + lg_volume) / beta + lg_ghs[beta]
                if lma < r1[k]:
                    r2[k] = lma
                    r2[k+1] = r1[k] + log(sqrt(1 - 1./beta), 2)
                    dec = (r1[k] - lma) + (r1[k+1] - r2[k+1])
                    for j in range(k+2, f):
                        r2[j] = r1[j] + dec/(beta - 2.)
                        t1[j] = True
                    phi = False

            for j in range(k, f):
                r1[j] = r2[j]

        if not any(t1):
            return r1, i + 1  # early termination

        # The last block looks HKZ-reduced, similar to [CN11].
        beta = min(45, block_size)
        lg_volume = sum(r1) - sum(r2[:-beta]) - sum(rk[-beta:])
        for k in range(d - beta, d):
            r2[k] = lg_volume / beta + rk[-(d - k)]
            t1[k] = True

        if r1 == r2:
            return r1, i + 1  # early termination
        r1 = copy(r2)
        t0 = copy(t1)

        if verbose:
            stats = {'i': i} | basis_quality([2.0**(2 * x) for x in r1])
            print(pretty_dict(OrderedDict(stats)))

    return r1, max_loops


def _extract_log_norms(r):
    if isinstance(r, IntegerMatrix):
        r = GSO.Mat(r)
    elif isinstance(r, MatGSO):
        r.update_gso()
        r = r.r()
    else:
        for ri in r:
            if ri <= 0:
                raise ValueError("squared norms in r should be positive")

    # code uses log2 of norms, FPLLL uses squared norms
    return list(map(lambda x: log(x, 2) / 2.0, r))


def simulate(r, param):
    """
    BKZ simulation algorithm as proposed by Chen and Nguyen [CN11].  This
    version terminates when no substantial progress is made anymore or when
    ``max_loops`` tours were simulated. If no ``max_loops`` is given, at most
    ``d`` tours are performed, where ``d`` is the dimension of the lattice.
    :param r: squared norms of the GSO vectors of the basis.
    :param param: BKZ parameters (block_size, max_loops, verbose)
    :returns: tuple with:
              1. the reduced squared norms of the GSO vectors of the basis,
              2. and the number of BKZ tours simulated.
    EXAMPLE::
        >>> from fpylll import IntegerMatrix, GSO, LLL, FPLLL, BKZ
        >>> FPLLL.set_random_seed(1337)
        >>> A = LLL.reduction(IntegerMatrix.random(100, "qary", bits=30, k=50))
        >>> M = GSO.Mat(A)
        >>> from fpylll.tools.bkz_simulator import simulate
        >>> _ = simulate(M, BKZ.Param(block_size=2, max_loops=4, flags=BKZ.VERBOSE))
        {"i":        0,  "r_0":   2^35.2,  "r_0/gh": 7.720730,  "rhf": 1.019532,  "/": -0.08309,  "hv/hv": 2.831284}
        {"i":        1,  "r_0":   2^35.2,  "r_0/gh": 7.513168,  "rhf": 1.019393,  "/": -0.08300,  "hv/hv": 2.828248}
        {"i":        2,  "r_0":   2^35.2,  "r_0/gh": 7.492349,  "rhf": 1.019379,  "/": -0.08291,  "hv/hv": 2.825635}
        {"i":        3,  "r_0":   2^35.1,  "r_0/gh": 7.427548,  "rhf": 1.019335,  "/": -0.08282,  "hv/hv": 2.822987}
        >>> _ = simulate(M, BKZ.Param(block_size=40, max_loops=4, flags=BKZ.VERBOSE))
        {"i":        0,  "r_0":   2^34.7,  "r_0/gh": 5.547855,  "rhf": 1.017849,  "/": -0.06928,  "hv/hv": 2.406481}
        {"i":        1,  "r_0":   2^34.2,  "r_0/gh": 3.894188,  "rhf": 1.016049,  "/": -0.06136,  "hv/hv": 2.150078}
        {"i":        2,  "r_0":   2^33.8,  "r_0/gh": 2.949459,  "rhf": 1.014638,  "/": -0.05735,  "hv/hv": 2.044402}
        {"i":        3,  "r_0":   2^33.6,  "r_0/gh": 2.574565,  "rhf": 1.013949,  "/": -0.05556,  "hv/hv": 1.999163}
    """
    profile, num_loops = log_simulate(
        _extract_log_norms(r), param.block_size, param.max_loops,
        param.flags & BKZ.VERBOSE,
    )
    return [2.0**(2 * x) for x in profile], num_loops


def simulate_prob(r, param, prng_seed=0xdeadbeef):
    """
    BKZ simulation algorithm as proposed by Bai, Stehlé and Wen [BSW18].  This
    version terminates when no substantial progress is made anymore or when
    ``max_loops`` tours were simulated. If no ``max_loops`` is given, at most
    ``d`` tours are performed, where ``d`` is the dimension of the lattice.
    :param r: squared norms of the GSO vectors of the basis.
    :param param: BKZ parameters (block_size, max_loops, verbose)
    :returns: tuple with:
              1. the reduced squared norms of the GSO vectors of the basis,
              2. and the number of BKZ tours simulated.
    EXAMPLE::
        >>> from fpylll import IntegerMatrix, GSO, LLL, FPLLL, BKZ
        >>> FPLLL.set_random_seed(1337)
        >>> A = LLL.reduction(IntegerMatrix.random(100, "qary", bits=30, k=50))
        >>> M = GSO.Mat(A)
        >>> from fpylll.tools.bkz_simulator import simulate_prob
        >>> _ = simulate_prob(M, BKZ.Param(block_size=40, max_loops=4, flags=BKZ.VERBOSE))
        {"i":        0,  "r_0":   2^34.5,  "r_0/gh": 4.714937,  "rhf": 1.017021,  "/": -0.06936,  "hv/hv": 2.410445}
        {"i":        1,  "r_0":   2^34.2,  "r_0/gh": 3.874259,  "rhf": 1.016023,  "/": -0.06189,  "hv/hv": 2.162205}
        {"i":        2,  "r_0":   2^33.8,  "r_0/gh": 2.996068,  "rhf": 1.014718,  "/": -0.05798,  "hv/hv": 2.056934}
        {"i":        3,  "r_0":   2^33.7,  "r_0/gh": 2.773499,  "rhf": 1.014326,  "/": -0.05598,  "hv/hv": 2.012050}
    """
    profile, num_loops = log_simulate_prob(
        _extract_log_norms(r), param.block_size, prng_seed, param.max_loops,
        param.flags & BKZ.VERBOSE,
    )
    return [2.0**(2 * x) for x in profile], num_loops


def averaged_simulate_prob(r, param, tries=10):
    """
    This wrapper calls the [BSW18] probabilistic BKZ simulator with different
    PRNG seeds, and returns the average output.
    Note: exp(E[ log(X_i) ]) is reported where X_i is the squared norm of the
    i-th GSO vector.
    :param r: squared norms of the GSO vectors of the basis.
    :param param: BKZ parameters (block_size, max_loops, verbose)
    :param tries: number of iterations to average over. Default: 10
    :returns: tuple with:
              1. averaged reduced squared norms of GSO vectors of the basis,
              2. the averaged number of BKZ tours simulated.
    EXAMPLE::
        >>> from fpylll import IntegerMatrix, GSO, LLL, FPLLL, BKZ
        >>> from fpylll.tools.bkz_simulator import averaged_simulate_prob
        >>> FPLLL.set_random_seed(1337)
        >>> A = LLL.reduction(IntegerMatrix.random(100, "qary", bits=30, k=50))
        >>> M = GSO.Mat(A)
        >>> _ = averaged_simulate_prob(M, BKZ.Param(block_size=40, max_loops=4))
        >>> print(_[0][:3])
        [13371442256.252..., 12239031147.433..., 12256303707.863...]
    """
    if tries < 1:
        raise ValueError("Need to average over positive number of tries.")
    profile = _extract_log_norms(r)

    profiles, loops = zip(*[log_simulate_prob(
        profile, param.block_size, i + 1, param.max_loops,
        param.flags & BKZ.VERBOSE
    ) for i in range(tries)])

    avg_profile = [sum(x) / len(x) for x in zip(*profiles)]
    avg_loops = sum(loops) / tries
    return [2.0**(2 * x) for x in avg_profile], avg_loops
