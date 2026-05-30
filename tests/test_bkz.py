# -*- coding: utf-8 -*-

from fpylll import GSO, IntegerMatrix, BKZ, LLL
from fpylll.config import float_types
from copy import copy

import tools

dimensions = ((0, 0), (1, 1), (2, 2), (3, 3), (10, 10), (50, 50), (60, 60),)


def make_integer_matrix(m, n, int_type="mpz"):
    A = IntegerMatrix(m, n, int_type=int_type)
    A.randomize("uniform", bits=20)
    return A


def test_bkz_init():
    for m, n in dimensions:
        A = make_integer_matrix(m, n)
        for float_type in float_types:
            M = GSO.Mat(copy(A), float_type=float_type)
            lll_obj = LLL.Reduction(M)
            param = BKZ.Param(block_size=3, strategies=BKZ.DEFAULT_STRATEGY)
            bkz = BKZ.Reduction(M, lll_obj, param)
            del bkz


def test_bkz_bkz():
    for m, n in dimensions:
        if m < 2 or n < 2:
            continue
        A = make_integer_matrix(m, n)
        b00 = []
        for float_type in float_types:
            B = copy(A)
            M = GSO.Mat(B, float_type=float_type)
            lll_obj = LLL.Reduction(M)
            param = BKZ.Param(block_size=min(m, 40), strategies=BKZ.DEFAULT_STRATEGY)
            bkz = BKZ.Reduction(M, lll_obj, param)
            bkz()
            b00.append(B[0, 0])
        for i in range(1, len(b00)):
            assert b00[0] == b00[i]


def test_bkz_gram_bkz_coherence():
    """
        Test if BKZ is coherent if it is given a matrix A or its associated
        Gram matrix A*A^T

        We should have Gram(BKZ_basis(A)) = BKZ_Gram(Gram(A)).
    """

    for m, n in dimensions:
        if m < 2 or n < 2:
            continue

        for float_type in float_types:
            A = make_integer_matrix(m, n)
            G = tools.compute_gram(A)

            GSO_A = GSO.Mat(A, float_type=float_type)
            GSO_G = GSO.Mat(G, float_type=float_type, gram=True)

            lll_obj_a = LLL.Reduction(GSO_A)
            lll_obj_g = LLL.Reduction(GSO_G)

            param = BKZ.Param(block_size=min(m, 40), strategies=BKZ.DEFAULT_STRATEGY)
            bkz_a = BKZ.Reduction(GSO_A, lll_obj_a, param)
            bkz_g = BKZ.Reduction(GSO_G, lll_obj_g, param)

            bkz_a()
            bkz_g()

            G_updated = tools.compute_gram(A)
            for i in range(m):
                for j in range(i + 1):
                    assert G_updated[i, j] == G[i, j]


def test_bkz_param_auto_abort_tuple():
    # try valid tuple
    param = BKZ.Param(block_size=10, auto_abort=(1.0, 2))
    assert param.flags & BKZ.AUTO_ABORT
    assert param.auto_abort == (1.0, 2)
    assert param.dict(False)["auto_abort"] == (1.0, 2)
    # try invalid tuple
    try:
        BKZ.Param(block_size=10, auto_abort=(1.0, 2, 3))
        assert False
    except ValueError:
        pass
    # try invalid value
    try:
        BKZ.Param(block_size=10, auto_abort=("not-a-float", 2))
        assert False
    except ValueError:
        pass
    try:
        BKZ.Param(block_size=10, auto_abort=(1.0, "not-a-int"))
        assert False
    except ValueError:
        pass


def test_bkz_param_gh_factor():
    param = BKZ.Param(block_size=10, gh_factor=1)
    assert param.flags & BKZ.GH_BND
    assert param.gh_factor == 1.0
    # when gh_factor is False, should not set flag
    param = BKZ.Param(block_size=10, gh_factor=False)
    assert not (param.flags & BKZ.GH_BND)
