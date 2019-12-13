# -*- coding: utf-8 -*-

from fpylll.fplll.gso import MatGSO
from fpylll.fplll.integer_matrix import IntegerMatrix
from fpylll.fplll.lll import LLLReduction
from fpylll.fplll.enumeration import Enumeration
from fpylll import Pruning
from time import time


def bench_enumeration(n):
    """Return number of nodes visited and wall time for enumeration in dimension `n`.

    :param n: dimension
    :returns: nodes, wall time

    >>> import fpylll.tools.benchmark
    >>> _ = fpylll.tools.benchmark.bench_enumeration(30)

    """

    A = IntegerMatrix.random(n, "qary", bits=30, k=n//2)
    M = MatGSO(A)
    L = LLLReduction(M)
    L(0, 0, n)

    radius = M.get_r(0, 0) * .999
    pruning = Pruning.run(radius, 2.0**50, M.r(), 0.2)

    enum = Enumeration(M)
    t = time()
    enum.enumerate(0, n, radius, 0, pruning=pruning.coefficients)
    t = time() - t
    cost = enum.get_nodes()

    return cost, t
