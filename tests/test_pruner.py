# -*- coding: utf-8 -*-

from fpylll import IntegerMatrix, GSO, LLL, prune

try:
    from fpylll.numpy import dump_r
    have_numpy = True
except ImportError:
    have_numpy = False


def prepare(n, m):
    A = [IntegerMatrix.random(n, "qary", bits=n, k=n) for _ in range(m)]
    M = [GSO.Mat(a) for a in A]
    L = [LLL.Reduction(m) for m in M]
    [l() for l in L]
    return M


def test_pruner_vec(n=20, m=20):
    M = prepare(n, m)
    if have_numpy:
        vec = []
        for m in M:
            vec.append(tuple(dump_r(m, 0, n)))

    radius = sum([mat.get_r(0, 0) for mat in M])/len(M)
    pruning = prune(None, radius, 0, 0.9, vec)
    assert pruning.expectation >= 0.89
    pruning = prune(pruning, radius, 0, 0.9, vec, reset=False)
    assert pruning.expectation >= 0.89
