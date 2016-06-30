# -*- coding: utf-8 -*-
include "fpylll/config.pxi"
include "cysignals/signals.pxi"

"""
Block Korkine Zolotarev algorithm.

..  moduleauthor:: Martin R.  Albrecht <martinralbrecht+fpylll@googlemail.com>
"""

from fplll cimport BKZ_MAX_LOOPS, BKZ_MAX_TIME, BKZ_DUMP_GSO, BKZ_DEFAULT
from fplll cimport BKZ_VERBOSE, BKZ_NO_LLL, BKZ_BOUNDED_LLL, BKZ_GH_BND, BKZ_AUTO_ABORT
from fplll cimport FloatType
from fplll cimport RED_BKZ_LOOPS_LIMIT, RED_BKZ_TIME_LIMIT
from fplll cimport bkzReduction
from fplll cimport getRedStatusStr

from fpylll.util import ReductionError

from bkz_params cimport BKZParam, BKZAutoAbort
from integer_matrix cimport IntegerMatrix
from fpylll.util cimport check_delta, check_precision, check_float_type


def bkz_reduction(IntegerMatrix B, BKZParam o, float_type=None, int precision=0):
    """
    Run BKZ reduction.

    :param IntegerMatrix B: Integer matrix, modified in place.
    :param BKZParam o: BKZ parameters
    :param float_type: either ``None``: for automatic choice or an entry of `fpylll.float_types`
    :param precision: bit precision to use if ``float_tpe`` is ``'mpfr'``

    :returns: modified matrix ``B``
    """
    check_precision(precision)

    cdef FloatType floatType = check_float_type(float_type)
    cdef int r = 0

    with nogil:
        sig_on()
        r = bkzReduction(B._core, NULL, o.o[0], floatType, precision)
        sig_off()

    if r and r not in (RED_BKZ_LOOPS_LIMIT, RED_BKZ_TIME_LIMIT):
        raise ReductionError( str(getRedStatusStr(r)) )

    return B

class BKZ:
    DEFAULT = BKZ_DEFAULT
    VERBOSE = BKZ_VERBOSE
    NO_LLL = BKZ_NO_LLL
    BOUNDED_LLL = BKZ_BOUNDED_LLL
    GH_BND = BKZ_GH_BND
    AUTO_ABORT = BKZ_AUTO_ABORT
    MAX_LOOPS = BKZ_MAX_LOOPS
    MAX_TIME = BKZ_MAX_TIME
    DUMP_GSO = BKZ_DUMP_GSO

    Param = BKZParam
    AutoAbort = BKZAutoAbort
    reduction = bkz_reduction
