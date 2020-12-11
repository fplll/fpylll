# -*- coding: utf-8 -*-

from fpylll import Enumeration, GSO, IntegerMatrix, LLL, Pruning
from fpylll.util import gaussian_heuristic
try:
    from time import process_time  # Python 3
except ImportError:
    from time import clock as process_time  # Python 2

dim_oh = ((40, 2**22), (41, 2**22), (50, 2**24), (51, 2**24))


def prepare(n):
    A = IntegerMatrix.random(n, "qary", bits=n/2, k=n/2)
    M = GSO.Mat(A)
    L = LLL.Reduction(M)
    L()
    return M


def test_pruner():

    # A dummyPruningParams.run to load tabulated values
    Pruning.run(5, 50, 10*[1.], .5)

    for (n, overhead) in dim_oh:

        print(" \n ~~~~ Dim %d \n" % n)

        M = prepare(n)
        r = [M.get_r(i, i) for i in range(n)]

        print(" \n GREEDY")
        radius = gaussian_heuristic(r) * 1.6
        print("pre-greedy radius %.4e" % radius)
        tt = process_time()
        pruning =Pruning.run(radius, overhead, r, 200, flags=Pruning.ZEALOUS, metric="solutions")
        print("Time %.4e"%(process_time() - tt))
        print("post-greedy radius %.4e" % radius)
        print(pruning)
        print("cost %.4e" % sum(pruning.detailed_cost))
        solutions = Enumeration(M, nr_solutions=10000).enumerate(0, n, radius, 0, pruning=pruning.coefficients)
        print(len(solutions))
        assert len(solutions)/pruning.expectation < 2
        assert len(solutions)/pruning.expectation > .2

        print(" \n GRADIENT \n")

        print("radius %.4e" % radius)
        tt = process_time()
        pruning = Pruning.run(radius, overhead, r, 200, flags=Pruning.GRADIENT, metric="solutions")
        print("Time %.4e"%(process_time() - tt))
        print(pruning)
        print("cost %.4e" % sum(pruning.detailed_cost))
        solutions = Enumeration(M, nr_solutions=10000).enumerate(0, n, radius, 0, pruning=pruning.coefficients)
        print(len(solutions))
        assert len(solutions)/pruning.expectation < 2
        assert len(solutions)/pruning.expectation > .2

        print(" \n HYBRID \n")

        print("radius %.4e" % radius)
        tt = process_time()
        pruning = Pruning.run(radius, overhead, r, 200, flags=Pruning.ZEALOUS, metric="solutions")
        print("Time %.4e"%(process_time() - tt))
        print(pruning)
        print("cost %.4e" % sum(pruning.detailed_cost))
        solutions = Enumeration(M, nr_solutions=10000).enumerate(0, n, radius, 0, pruning=pruning.coefficients)
        print(len(solutions))
        assert len(solutions)/pruning.expectation < 2
        assert len(solutions)/pruning.expectation > .2
