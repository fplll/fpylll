# -*- coding: utf-8 -*-
include "fpylll/config.pxi"
include "cysignals/signals.pxi"


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

