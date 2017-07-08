# -*- coding: utf-8 -*-

from fpylll import get_precision, set_precision


def test_precision():
    set_precision(53)
    assert get_precision() == 53
    assert set_precision(100) == 53
    assert set_precision(100) == 100
