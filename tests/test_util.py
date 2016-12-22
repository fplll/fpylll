# -*- coding: utf-8 -*-

from copy import copy

from fpylll import LLL, IntegerMatrix, GSO
from fpylll.algorithms.util import gaussian_heuristic as gaussian_heuristic_py
from fpylll.util import gaussian_heuristic, set_random_seed
from fpylll.numpy import dump_r
from math import log, exp

dimensions = (20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 140, 200, 300, 400)


def make_integer_matrix(n):
    A = IntegerMatrix.random(n, "uniform", bits=10)
    return A


def test_gh():
    for n in dimensions:
        set_random_seed(n)
        A = make_integer_matrix(n)
        M = GSO.Mat(A, float_type="qd")
        M.discover_all_rows()
        lll = LLL.Reduction(M)
        lll()
        radius = M.get_r(0,0)
        root_det = M.get_root_det(0, n)
        print n
        print root_det
        gh_radius, ge = gaussian_heuristic(2000*radius, 0, n, root_det, 1.0)

        gh1 = gh_radius * 2**ge

        r = dump_r(M, 0, n)
        print exp(sum([log(x) for x in r])/n)
        gh2 = gaussian_heuristic_py(r)

        print gh1
        print gh2
        print
        assert abs(gh1/gh2 -1) < 0.01

test_gh()