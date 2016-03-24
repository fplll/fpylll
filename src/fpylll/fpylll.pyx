# -*- coding: utf-8 -*-
include "config.pxi"
include "cysignals/signals.pxi"


IF HAVE_QD:
    have_qd = True
    float_types = ("d", "ld", "dpe", "dd", "qd", "mpfr")
ELSE:
    have_qd = False
    float_types = ("d", "ld", "dpe", "mpfr")

class ReductionError(RuntimeError):
    pass
