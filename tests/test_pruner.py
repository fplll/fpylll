# -*- coding: utf-8 -*-

from fpylll import IntegerMatrix, GSO, LLL, prune, Enumeration
from fpylll.util import gaussian_heuristic

dim_oh = ((40, 2**22), (41, 2**22), (50, 2**24), (51, 2**24))

try:
    from fpylll.numpy import dump_r
    have_numpy = True
except ImportError:
    have_numpy = False


def prepare(n):
    A = IntegerMatrix.random(n, "qary", bits=n/2, k=n/2)
    M = GSO.Mat(A)
    L = LLL.Reduction(M)
    L()
    return M


def test_pruner_vec():
    for (n, overhead) in dim_oh:

        print " \n ==== Dim \n", n

        M = prepare(n)
        r = [M.get_r(i, i) for i in range(n)]

        print " \n GREEDY"
        radius = gaussian_heuristic(r) * 1.6
        print "pre-greedy radius %.4e" % radius
        (radius, pruning) = prune(radius, overhead, 200, r,
                                  descent_method="greedy", metric="solutions")
        print "post-greedy radius %.4e" % radius
        print pruning
        print "cost %.4e" % sum(pruning.detailed_cost)
        solutions = Enumeration(M, nr_solutions=10000).enumerate(0, n, radius, 0, pruning=pruning.coefficients)
        print len(solutions)
        assert len(solutions)/pruning.expectation < 2
        assert len(solutions)/pruning.expectation > .2

        print " \n GREEDY \n"
        print "pre-greedy radius %.4e" % radius
        (radius, pruning) = prune(radius, overhead, 200, r, descent_method="greedy", metric="solutions")
        print "post-greedy radius %.4e" % radius
        print pruning
        print "cost %.4e" % sum(pruning.detailed_cost)
        solutions = Enumeration(M, nr_solutions=10000).enumerate(0, n, radius, 0, pruning=pruning.coefficients)
        print len(solutions)
        assert len(solutions)/pruning.expectation < 2
        assert len(solutions)/pruning.expectation > .2

        print " \n GRADIENT \n"

        print "radius %.4e" % radius
        pruning = prune(radius, overhead, 200, r, descent_method="gradient", metric="solutions")
        print pruning
        #print pruning.coefficients
        print "cost %.4e" % sum(pruning.detailed_cost)
        solutions = Enumeration(M, nr_solutions=10000).enumerate(0, n, radius, 0, pruning=pruning.coefficients)
        print len(solutions)
        assert len(solutions)/pruning.expectation < 2
        assert len(solutions)/pruning.expectation > .2

        print " \n HYBRID \n"

        print "radius %.4e" % radius
        pruning = prune(radius, overhead, 200, r, descent_method="hybrid", metric="solutions")
        print pruning
        #print pruning.coefficients
        print "cost %.4e" % sum(pruning.detailed_cost)
        solutions = Enumeration(M, nr_solutions=10000).enumerate(0, n, radius, 0, pruning=pruning.coefficients)
        print len(solutions)
        assert len(solutions)/pruning.expectation < 2
        assert len(solutions)/pruning.expectation > .2


test_pruner_vec()
