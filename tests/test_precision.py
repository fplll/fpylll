# -*- coding: utf-8 -*-

from fpylll import FPLLL


def test_precision():
    FPLLL.set_precision(53)
    assert FPLLL.get_precision() == 53
    assert FPLLL.set_precision(100) == 53
    assert FPLLL.set_precision(100) == 100
