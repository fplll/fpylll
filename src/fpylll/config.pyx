# -*- coding: utf-8 -*-
include "fpylll/config.pxi"
include "cysignals/signals.pxi"


from fplll.fplll cimport default_strategy as default_strategy_c
from fplll.fplll cimport default_strategy_path as default_strategy_path_c

IF HAVE_QD:
    have_qd = True
    float_types = ("d", "ld", "dpe", "dd", "qd", "mpfr")
ELSE:
    have_qd = False
    float_types = ("d", "ld", "dpe", "mpfr")

IF HAVE_SAGE:
    have_sage = True
ELSE:
    have_sage = False

default_strategy = default_strategy_c().c_str()
default_strategy_path = default_strategy_path_c().c_str()
