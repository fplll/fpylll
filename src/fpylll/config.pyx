# -*- coding: utf-8 -*-
include "fpylll/config.pxi"


from .fplll.fplll cimport default_strategy as default_strategy_c
from .fplll.fplll cimport default_strategy_path as default_strategy_path_c

IF HAVE_LONG_DOUBLE:
    have_long_double = True
    float_types = ("d", "ld")
ELSE:
    have_long_double = False
    float_types = ("d",)

IF HAVE_QD:
    have_qd = True
    float_types = float_types + ("dpe", "dd", "qd", "mpfr")
ELSE:
    have_qd = False
    float_types = float_types + ("dpe", "mpfr")

int_types = ("long", "mpz")

default_strategy = default_strategy_c().c_str()
default_strategy_path = default_strategy_path_c().c_str()
