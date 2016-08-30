# -*- coding: utf-8 -*-

from fpylll.fplll.gso import MatGSO
from fpylll.fplll.integer_matrix import IntegerMatrix
from fpylll.fplll.lll import LLLReduction
from fpylll.fplll.enumeration import Enumeration
from fpylll.fplll.pruner import prune
from time import time


def bench_enumeration(n):
    """Return number of nodes visited and wall time for enumeration in dimension `n`.

    :param n: dimension
    :returns: nodes, wall time

    """

    A = IntegerMatrix.random(n, "qary", bits=5*n, k=1)
    M = MatGSO(A)
    L = LLLReduction(M)
    L(0, 0, n)

    radius = M.get_r(0, 0) * .999
    pruning = prune(radius, 2**30, 0.9, M)

    enum = Enumeration(M)
    t = time()
    enum.enumerate(0, n, radius, 0, pruning.coefficients)
    t = time() - t
    cost = enum.get_nodes()

    return cost, t
