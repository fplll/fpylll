# -*- coding: utf-8 -*-
include "fpylll/config.pxi"


from fpylll.fplll.decl cimport fp_nr_t
from fpylll.fplll.fplll cimport FP_NR, RandGen, dpe_t
from fpylll.fplll.fplll cimport FT_DEFAULT, FT_DOUBLE, FT_LONG_DOUBLE, FT_DPE, FT_MPFR
from fpylll.fplll.fplll cimport IntType, ZT_LONG, ZT_MPZ
from fpylll.fplll.fplll cimport adjust_radius_to_gh_bound as adjust_radius_to_gh_bound_c
from fpylll.fplll.fplll cimport set_external_enumerator as set_external_enumerator_c
from fpylll.fplll.fplll cimport extenum_fc_enumerate
from fpylll.fplll.fplll cimport get_root_det as get_root_det_c
from fpylll.fplll.fplll cimport PRUNER_METRIC_PROBABILITY_OF_SHORTEST, PRUNER_METRIC_EXPECTED_SOLUTIONS, PrunerMetric
from fpylll.fplll.gso cimport MatGSO
from fpylll.gmp.random cimport gmp_randstate_t, gmp_randseed_ui, gmp_urandomm_ui
from fpylll.mpfr.mpfr cimport mpfr_t
from math import log, exp, lgamma, pi
from libcpp.functional cimport function

IF HAVE_QD:
    from fpylll.qd.qd cimport dd_real, qd_real
    from fpylll.fplll.fplll cimport FT_DD, FT_QD

cdef extern from "util_helper.h":
    function[extenum_fc_enumerate] void_ptr_to_function(void *ptr)


float_aliases = {'d': 'double',
                 'ld': 'long double'}

# We return `object` to permit exceptions

cdef object check_float_type(object float_type):

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

cdef object check_int_type(object int_type):

    if int_type == "default" or int_type is None:
        return ZT_MPZ
    if int_type == "mpz":
        return ZT_MPZ
    if int_type == "long":
        return ZT_LONG

    raise ValueError("Integer type '%s' unknown." % int_type)

cdef PrunerMetric check_pruner_metric(object metric):
    if metric == "probability" or metric == PRUNER_METRIC_PROBABILITY_OF_SHORTEST:
        return PRUNER_METRIC_PROBABILITY_OF_SHORTEST
    elif metric == "solutions" or metric == PRUNER_METRIC_EXPECTED_SOLUTIONS:
        return PRUNER_METRIC_EXPECTED_SOLUTIONS
    else:
        raise ValueError("Pruner metric '%s' not supported."%metric)

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
    if not RandGen.get_initialized():
        RandGen.init()

    cdef gmp_randstate_t state = RandGen.get_gmp_state()
    gmp_randseed_ui(state, seed)

def randint(a, b):
    """
    Return random integer in range [a, b], including both end points.
    """
    if not RandGen.get_initialized():
        RandGen.init()
    cdef gmp_randstate_t state = RandGen.get_gmp_state()
    return (<long>gmp_urandomm_ui(state, b+1-a)) + a


def get_precision(float_type="mpfr"):
    """Get currently set precision

    :param float_type: one of 'double', 'long double', 'dpe', 'dd', 'qd' or 'mpfr'
    :returns: precision in bits

    This function returns the precision per type::

        >>> import fpylll
        >>> from fpylll import FPLLL
        >>> FPLLL.get_precision('double')
        53
        >>> if fpylll.config.have_long_double:
        ...     FPLLL.get_precision('long double')
        ... else:
        ...     64
        64
        >>> FPLLL.get_precision('dpe')
        53

    For the MPFR type different precisions are supported::

        >>> _ = FPLLL.set_precision(212)
        >>> FPLLL.get_precision('mpfr')
        212
        >>> FPLLL.get_precision()
        212
        >>> _ = FPLLL.set_precision(53)

    """
    cdef FloatType float_type_ = check_float_type(float_type)

    if float_type_ == FT_DOUBLE:
        return FP_NR[double].get_prec()
    IF HAVE_LONG_DOUBLE:
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


def adjust_radius_to_gh_bound(double dist, int dist_expo, int block_size, double root_det, double gh_factor):
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
    adjust_radius_to_gh_bound_c[FP_NR[double]](gh_dist, dist_expo, block_size, root_det_, gh_factor)
    return gh_dist.get_d(), dist_expo

class ReductionError(RuntimeError):
    pass


def ball_log_vol(n):
    """
    Return volume of `n`-dimensional unit ball

    :param n: dimension

    """
    return (n/2.) * log(pi) - lgamma(n/2. + 1)

def gaussian_heuristic(r):
    """
    Return squared norm of shortest vector as predicted by the Gaussian heuristic.

    :param r: vector of squared Gram-Schmidt norms

    """
    n = len(list(r))
    log_vol = sum([log(x) for x in r])
    log_gh =  1./n * (log_vol - 2 * ball_log_vol(n))
    return exp(log_gh)

cpdef set_external_enumerator(enumerator):
    """
    Set an external enumeration library.

    For example, assume you compiled a `fplll-extenum
    <https://github.com/cr-marcstevens/fplll-extenum>`_

    First, we load the required Python modules: fpylll and `ctypes
    <https://docs.python.org/2/library/ctypes.html>`_

    >>> from fpylll import *  # doctest: +SKIP
    >>> import ctypes         # doctest: +SKIP

    Then, using ``ctypes`` we dlopen ``enumlib.so``

    >>> enumlib = ctypes.cdll.LoadLibrary("enumlib.so") # doctest: +SKIP

    For demonstration purposes we increase the loglevel. Note that functions names are result of C++
    compiler name mangling and may differ depending on platform/compiler/linker.

    >>> enumlib._Z20enumlib_set_logleveli(1)            # doctest: +SKIP

    We grab the external enumeration function

    >>> fn = enumlib._Z17enumlib_enumerateidSt8functionIFvPdmbS0_S0_EES_IFddS0_EES_IFvdS0_iEEbb # doctest: +SKIP

    and pass it to Fplll

    >>> FPLLL.set_external_enumerator(fn)  # doctest: +SKIP

    To disable the external enumeration library, call

    >>> FPLLL.set_external_enumerator(None)  # doctest: +SKIP

    """
    import ctypes
    cdef unsigned long p
    if not enumerator:
        set_external_enumerator_c(<function[extenum_fc_enumerate]>NULL)
    elif isinstance(enumerator, ctypes._CFuncPtr):
        p = ctypes.cast(enumerator, ctypes.c_void_p).value
        set_external_enumerator_c(void_ptr_to_function(<void *>p))

class FPLLL:
    set_precision = staticmethod(set_precision)
    get_precision = staticmethod(get_precision)
    set_random_seed = staticmethod(set_random_seed)
    randint = staticmethod(randint)
    set_external_enumerator = staticmethod(set_external_enumerator)
