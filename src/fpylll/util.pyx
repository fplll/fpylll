# -*- coding: utf-8 -*-
include "fpylll/config.pxi"
include "cysignals/signals.pxi"


from fpylll.fplll.decl cimport mpz_double, mpz_ld, mpz_dpe, mpz_mpfr, fp_nr_t
from fpylll.fplll.fplll cimport FP_NR, RandGen, dpe_t
from fpylll.fplll.fplll cimport FT_DEFAULT, FT_DOUBLE, FT_LONG_DOUBLE, FT_DPE, FT_MPFR
from fpylll.fplll.fplll cimport gaussian_heuristic as gaussian_heuristic_c
from fpylll.fplll.fplll cimport get_root_det as get_root_det_c
from fpylll.fplll.fplll cimport PRUNER_METHOD_GRADIENT, PRUNER_METHOD_NM, PRUNER_METHOD_HYBRID
from fpylll.fplll.gso cimport MatGSO
from fpylll.gmp.random cimport gmp_randstate_t, gmp_randseed_ui
from fpylll.mpfr.mpfr cimport mpfr_t

IF HAVE_QD:
    from fpylll.qd.qd cimport dd_real, qd_real
    from fpylll.fplll.fplll cimport FT_DD, FT_QD


float_aliases = {'d': 'double',
                 'ld': 'long double'}

cdef FloatType check_float_type(object float_type):

    float_type = float_aliases.get(float_type, float_type)

    if float_type == "default" or float_type is None:
        return FT_DEFAULT
    if float_type == "double":
        return FT_DOUBLE
    if float_type == "long double":
        return FT_LONG_DOUBLE
    if float_type == "dpe":
        return FT_DPE
    IF HAVE_QD:
        if float_type == "dd":
            return FT_DD
        if float_type == "qd":
            return FT_QD
    if float_type == "mpfr":
        return FT_MPFR

    raise ValueError("Float type '%s' unknown." % float_type)

cdef int check_descent_method(object descent_method) except -1:
    if descent_method == "gradient":
        return PRUNER_METHOD_GRADIENT
    if descent_method == "nm":
        return PRUNER_METHOD_NM
    if descent_method == "hybrid":
        return PRUNER_METHOD_HYBRID
    else:
        raise ValueError("Descent method '%s' not supported."%descent_method)

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
    if precision < 53 and precision != 0:
        raise TypeError("precision must be >= 53 or equal to 0")

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
    cdef gmp_randstate_t state = RandGen.get_gmp_state()
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

    For the MPFR type different precisions are supported::

        >>> _ = set_precision(212)
        >>> get_precision('mpfr')
        212
        >>> get_precision()
        212
        >>> _ = set_precision(53)

    """
    cdef FloatType float_type_ = check_float_type(float_type)

    if float_type_ == FT_DOUBLE:
        return FP_NR[double].get_prec()
    if float_type_ == FT_LONG_DOUBLE:
        return FP_NR[longdouble].get_prec()
    if float_type_ == FT_DPE:
        return FP_NR[dpe_t].get_prec()
    IF HAVE_QD:
        if float_type_ == FT_DD:
            return FP_NR[dd_real].get_prec()
        if float_type_ == FT_QD:
            return FP_NR[qd_real].get_prec()
    if float_type_ == FT_MPFR:
        return FP_NR[mpfr_t].get_prec()
    raise ValueError("Floating point type '%s' unknown."%float_type)

def set_precision(unsigned int prec):
    """Set precision globally for MPFR

    :param prec: an integer >= 53
    :returns: current precision

    """
    if prec == 0:
        prec = 53
    if prec < 53:
        raise ValueError("Precision (%d) too small."%prec)
    return FP_NR[mpfr_t].set_prec(prec)

class PrecisionContext:
    def __init__(self, prec):
        """Create new precision context.

        :param prec: internal precision

        """
        self.prec = prec

    def __enter__(self):
        self.prec = set_precision(self.prec)

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.prec = set_precision(self.prec)

def precision(prec):
    """Create new precision context.

    :param prec: internal precision

    """
    return PrecisionContext(prec)


def gaussian_heuristic(double dist, int dist_expo, int block_size, double root_det, double gh_factor):
    """
    Use Gaussian Heuristic to reduce bound on the length of the shortest vector.

    :param double dist: norm of shortest vector
    :param int dist_expo: exponent of norm (for dpe representation)
    :param int block_size: block size
    :param double root_det: root determinant
    :param double gh_factor: factor to multiply with

    :returns: (dist, expo)

    """
    cdef FP_NR[double] gh_dist = dist
    cdef FP_NR[double] root_det_ = root_det
    gaussian_heuristic_c[FP_NR[double]](gh_dist, dist_expo, block_size, root_det_, gh_factor)
    return gh_dist.get_d(), dist_expo

class ReductionError(RuntimeError):
    pass
