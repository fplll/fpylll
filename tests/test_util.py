# -*- coding: utf-8 -*-

from fpylll import IntegerMatrix, GSO
from fpylll.util import adjust_radius_to_gh_bound, set_random_seed, gaussian_heuristic

dimensions = [20, 21, 40, 41, 60, 61, 80, 81, 100, 101, 200, 201, 300, 301, 400, 401]


def make_integer_matrix(n):
    A = IntegerMatrix.random(n, "uniform", bits=30)
    return A


def test_gh():
    try:
        from fpylll.numpy import dump_r
    except ImportError:
        return

    for n in dimensions:
        set_random_seed(n)
        A = make_integer_matrix(n)
        try:
            M = GSO.Mat(A, float_type="ld")
        except ValueError:
            M = GSO.Mat(A, float_type="d")
        M.discover_all_rows()
        M.update_gso()
        radius = M.get_r(0, 0)
        root_det = M.get_root_det(0, n)
        gh_radius, ge = adjust_radius_to_gh_bound(2000*radius, 0, n, root_det, 1.0)

        gh1 = gh_radius * 2**ge

        r = dump_r(M, 0, n)
        gh2 = gaussian_heuristic(r)
        assert abs(gh1/gh2 -1) < 0.01
