# -*- coding: utf-8 -*-
include "fpylll/config.pxi"
include "cysignals/signals.pxi"

"""
Block Korkine Zolotarev algorithm.

..  moduleauthor:: Martin R.  Albrecht <martinralbrecht+fpylll@googlemail.com>
"""

from bkz_params cimport BKZParam
from decl cimport mpz_double, mpz_mpfr
from fplll cimport BKZAutoAbort as BKZAutoAbort_c
from fplll cimport BKZ_MAX_LOOPS, BKZ_MAX_TIME, BKZ_DUMP_GSO, BKZ_DEFAULT
from fplll cimport BKZ_VERBOSE, BKZ_NO_LLL, BKZ_BOUNDED_LLL, BKZ_GH_BND, BKZ_AUTO_ABORT
from fplll cimport FP_NR, Z_NR
from fplll cimport FloatType
from fplll cimport RED_BKZ_LOOPS_LIMIT, RED_BKZ_TIME_LIMIT
from fplll cimport bkzReduction
from fplll cimport getRedStatusStr
from fpylll.gmp.mpz cimport mpz_t
from fpylll.mpfr.mpfr cimport mpfr_t
from fpylll.util cimport check_delta, check_precision, check_float_type
from fpylll.util import ReductionError
from integer_matrix cimport IntegerMatrix


cdef class BKZAutoAbort:
    """
    """
    def __init__(self, MatGSO M, int num_rows, int start_row=0):
        """FIXME! briefly describe function

        :param MatGSO M:
        :param int num_rows:
        :param int start_row:
        :returns:
        :rtype:

        """
        if M._type == mpz_double:
            self._type = mpz_double
            self._core.mpz_double = new BKZAutoAbort_c[FP_NR[double]](M._core.mpz_double[0],
                                                                      num_rows,
                                                                      start_row)
        elif M._type == mpz_mpfr:
            self._type = mpz_mpfr
            self._core.mpz_mpfr = new BKZAutoAbort_c[FP_NR[mpfr_t]](M._core.mpz_mpfr[0],
                                                                  num_rows,
                                                                  start_row)
        else:
            raise RuntimeError("BKZAutoAbort object '%s' has no core."%self)

        self.M = M

    def test_abort(self, scale=1.0, int max_no_dec=5):
        """FIXME! briefly describe function

        :param scale:
        :param int max_no_dec:
        :returns:
        :rtype:

        """
        if self._type == mpz_double:
            return self._core.mpz_double.test_abort(scale, max_no_dec)
        elif self._type == mpz_mpfr:
            return self._core.mpz_mpfr.test_abort(scale, max_no_dec)
        else:
            raise RuntimeError("BKZAutoAbort object '%s' has no core."%self)


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
