# -*- coding: utf-8 -*-
from fpylll import FPLLL, IntegerMatrix, LLL, GSO, Enumeration


def test_callback_enum(d=40):

    FPLLL.set_random_seed(0x1337)
    A = LLL.reduction(IntegerMatrix.random(100, "qary", k=50, q=7681))
    M = GSO.Mat(A)
    M.update_gso()

    # we are not imposing a constraint
    enum_obj = Enumeration(M)
    solutions = enum_obj.enumerate(0, d, 0.99 * M.get_r(0, 0), 0)
    max_dist, sol = solutions[0]
    assert A.multiply_left(sol)[0] != 2

    # now we do
    def callback(new_sol_coord):
        if A.multiply_left(new_sol_coord)[0] == 2:
            return True
        else:
            return False

    enum_obj = Enumeration(M, callbackf=callback)
    solutions = enum_obj.enumerate(0, d, 0.99 * M.get_r(0, 0), 0)
    max_dist, sol = solutions[0]

    assert A.multiply_left(sol)[0] == 2
