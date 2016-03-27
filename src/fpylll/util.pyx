# -*- coding: utf-8 -*-
include "config.pxi"
include "cysignals/signals.pxi"


from fplll cimport FT_DEFAULT, FT_DOUBLE, FT_LONG_DOUBLE, FT_DD, FT_QD, FT_DPE, FT_MPFR
from fplll cimport FP_NR, RandGen, dpe_t
from qd.qd cimport dd_real, qd_real
from fpylll.gmp.random cimport gmp_randstate_t, gmp_randseed_ui
from fpylll.mpfr.mpfr cimport mpfr_t

float_aliases = {'d': 'double',
                 'ld': 'long double'}

cdef FloatType check_float_type(object float_type):
    cdef FloatType float_type_

    float_type = float_aliases.get(float_type, float_type)

    if float_type == "default" or float_type is None:
        float_type_= FT_DEFAULT
    elif float_type == "double":
        float_type_ = FT_DOUBLE
    elif float_type == "long double":
        float_type_ = FT_LONG_DOUBLE
    elif float_type == "dd":
        float_type_ = FT_DD
    elif float_type == "qd":
        float_type_ = FT_QD
    elif float_type == "dpe":
        float_type_ = FT_DPE
    elif float_type == "mpfr":
        float_type_ = FT_MPFR
    else:
        raise ValueError("Float type '%s' unknown." % float_type)
    return float_type_

cdef int preprocess_indices(int &i, int &j, int m, int n) except -1:
    if i < 0:
        (&i)[0] %= m
    if j < 0:
        (&j)[0] %= n

    if i >= m:
        raise IndexError("First index must be < %d but got %d."%(n, i))
    if j >= n:
        raise IndexError("Second index must be < %d but got %d."%(n, j))

    return 0

cdef int check_precision(int precision) except -1:
    """
    Check whether the provided precision is within valid bounds. If not raise a ``TypeError``.

     :param precision: an integer
    """
    if precision < 0:
        raise TypeError("precision must be >= 0")

cdef int check_eta(float eta) except -1:
    """
    Check whether the provided parameter ``eta`` is within valid bounds. If not raise a ``TypeError``.

     :param eta: a floating point number
    """
    if eta < 0.5:
        raise TypeError("eta must be >= 0.5")

cdef int check_delta(float delta) except -1:
    """
    Check whether the provided parameter ``delta`` is within valid bounds.  If not raise a
    ``TypeError``.

    :param delta: a floating point number
    """
    if delta <= 0.25:
        raise TypeError("delta must be > 0.25")
    elif delta > 1.0:
        raise TypeError("delta must be <= 1.0")


def set_random_seed(unsigned long seed):
    """Set random seed.

    :param seed: a new seed.

    """
    cdef gmp_randstate_t state = RandGen.getGMPState()
    gmp_randseed_ui(state, seed)

def get_precision(float_type="mpfr"):
    """Get currently set precision

    :param float_type: one of 'double', 'long double', 'dpe', 'dd', 'qd' or 'mpfr'
    :returns: precision in bits

    This function returns the precision per type::

        >>> from fpylll import get_precision, set_precision
        >>> get_precision('double')
        53
        >>> get_precision('long double')
        64
        >>> get_precision('dpe')
        53
        >>> get_precision('dd')
        106
        >>> get_precision('qd')
        212

    For the MPFR type different precisions are supported::

        >>> _ = set_precision(212)
        >>> get_precision('mpfr')
        212
        >>> get_precision()
        212

    """
    cdef FloatType float_type_ = check_float_type(float_type)

    if float_type_ == FT_DOUBLE:
        return FP_NR[double].getprec()
    elif float_type_ == FT_LONG_DOUBLE:
        return FP_NR[longdouble].getprec()
    elif float_type_ == FT_DPE:
        return FP_NR[dpe_t].getprec()
    elif float_type_ == FT_DD:
        return FP_NR[dd_real].getprec()
    elif float_type_ == FT_QD:
        return FP_NR[qd_real].getprec()
    elif float_type_ == FT_MPFR:
        return FP_NR[mpfr_t].getprec()
    else:
        raise ValueError("Floating point type '%s' unknown."%float_type)

def set_precision(unsigned int prec):
    """Set precision globally for MPFR

    :param prec: an integer >= 53
    :returns: current precision

    """
    if prec < 53:
        raise ValueError("Precision (%d) too small."%prec)
    return FP_NR[mpfr_t].setprec(prec)
