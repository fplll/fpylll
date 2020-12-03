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
    elif isinstance(r, MatGSO):
        r.update_gso()
        r = r.r()
    else:
        for ri in r:
            if (ri <= 0):
                raise ValueError("squared norms in r should be positive")
            
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

    assert (param.block_size >= 3) 
    
    if not prng_seed:
        prng_seed = FPLLL.randint(0, 2**32-1)

    random.seed(prng_seed)

    if isinstance(r, IntegerMatrix):
        r = GSO.Mat(r)
    elif isinstance(r, MatGSO):
        r.update_gso()
        r = r.r()
    else:
        for ri in r:
            if (ri <= 0):
                raise ValueError("squared norms in r should be positive")
            
    n = len(r)
    assert (n >= 45)  # I didn't check this, presumably this is not needed in any case.
    
    # code uses log2 of norms, FPLLL uses squared norms
    r = list(map(lambda x: log(x, 2) / 2.0, r))
    l = copy(r)
    l2 = copy(r)
    c = [rk[-j] - sum(rk[-j:]) / j for j in range(1, 46)]
    c += [
        (lgamma(d / 2.0 + 1) * (1.0 / d) - log(sqrt(pi))) / log(2.0)
        for d in range(46, param.block_size + 1)
    ]
        
    if param.max_loops:
        N = param.max_loops
    else:
        N = param.block_size
        
    t0 = [True for _ in range(n)]
    for ntours in range(N):
        
        t1 = [False for _ in range(n)]
        for k in range(n - min(45, param.block_size)):
            bs = min(param.block_size, n - k)
            e = k + bs
            tau = False
            for kp in range(k, e):
                tau |= t0[kp]
            logdet = sum(l[:e]) - sum(l2[:k])
            if tau:
                X = random.expovariate(.5)
                g = (log(X, 2) + logdet) / bs + c[bs - 1]
                if g < l[k]:
                    l2[k] = g
                    l2[k+1] = l[k] + log(sqrt(1-1./bs), 2)
                    dec = (l[k]-g) + (l[k+1] - l2[k+1])
                    for j in range(k+2, e):
                        l2[j] = l[j] + dec/(bs-2.)
                        t1[j] = True
                    tau = False
                    
            for j in range(k, e):
                l[j] = l2[j]

        # early termination
        if True not in t1:
            break
                
        # last block
        d = min(45, param.block_size)
        logdet = sum(l) - sum(l2[:-d])
        if param.block_size < 45:
            rk1 = normalize_GSO_unitary(rk[-d:])
        else:
            rk1 = rk
        K = range(n-d,n)
        for k, r in zip(K, rk1):
            l2[k] = logdet / d + r
            t1[k] = True

        # early termination
        if (l == l2):
            break
        l = copy(l2)
        t0 = copy(t1)
            
        if param.flags & BKZ.VERBOSE:
            r = OrderedDict()
            r["i"] = ntours
            for k, v in basis_quality(list(map(lambda x: 2.0 ** (2 * x), l))).items():
                r[k] = v
            print(pretty_dict(r))

    
    l = list(map(lambda x: 2.0 ** (2 * x), l))
    return l, ntours + 1



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
        >>> print(_)
        ([4663149828.487716, 4267813469.1884093, 4273411937.5775056, 3990268150.7410364, 3668046500.478677, 3553844200.8006516, 3444658019.03044, 3156147632.004578, 2968314152.3642693, 2842135617.896822, 2701726280.1863375, 2427431897.697585, 2295859412.824674, 2216333203.90082, 2074499322.5704694, 1924603314.4880495, 1827814549.0131829, 1776013797.9678044, 1664207295.6809466, 1543482438.1355932, 1455023148.1512935, 1379142479.9794054, 1356871071.2347507, 1194384683.735174, 1137077656.9115803, 1112097473.6316075, 1033423233.359938, 980213545.1737213, 942855227.161819, 895255072.7324882, 789627704.6799444, 801073824.8874698, 755775217.5311928, 713222385.7553864, 692363947.354517, 639206877.5724745, 582893859.6217077, 580700466.2116839, 513842580.5500117, 522884517.55556744, 472608694.88618803, 463127438.182499, 425511897.0308301, 405984855.563072, 391768431.9088576, 360207598.4480099, 346098972.17481536, 319040929.6867722, 307763348.5992211, 289341631.5133158, 272510080.6920898, 272034238.24068666, 249216270.5489254, 232794295.48157167, 219610297.87482646, 199455041.13201967, 193348329.31101698, 185461184.88176084, 173040214.2061931, 157768256.75544098, 153612053.69867587, 146487055.0586665, 137667229.3463833, 132369126.46309029, 126562141.62680627, 120933372.29843551, 116256121.52541411, 109239936.60474023, 105984052.87368423, 98377786.19612649, 94040018.4264894, 90144171.47857843, 85359372.13250272, 80697530.2886767, 76878503.96829975, 71698458.78848211, 67542901.8178413, 63851179.487097755, 59514031.1756667, 58809384.563065544, 53967304.511024304, 51405889.70498139, 48106140.863982655, 45990641.71350795, 42528083.76302789, 40617154.94932017, 37881005.972814575, 36283044.50300712, 33995717.68532665, 31827249.439673737, 29482737.620986808, 28223587.79306287, 26319539.552730706, 24319773.77137055, 22856719.79128099, 21615530.795365877, 20040693.668807715, 18664350.946514044, 18039907.11784504, 17985296.927018106], 4.0)
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
            inew = [sum(l) for l in zip(i,x)]
            i = inew
            j += y

    i = [l/tries for l in i]
    j = j/tries
    return list(map(exp, i)), j
