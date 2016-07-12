# -*- coding: utf-8 -*-
include "fpylll/config.pxi"
include "cysignals/signals.pxi"


from fpylll.fplll.decl cimport mpz_double, mpz_ld, mpz_dpe, mpz_mpfr, fp_nr_t
from fpylll.fplll.fplll cimport FP_NR, RandGen, dpe_t
from fpylll.fplll.fplll cimport FT_DEFAULT, FT_DOUBLE, FT_LONG_DOUBLE, FT_DPE, FT_MPFR
from fpylll.fplll.fplll cimport gaussian_heuristic as gaussian_heuristic_c
from fpylll.fplll.fplll cimport get_root_det as get_root_det_c
from fpylll.fplll.gso cimport MatGSO
from fpylll.gmp.random cimport gmp_randstate_t, gmp_randseed_ui
from fpylll.mpfr.mpfr cimport mpfr_t

IF HAVE_QD:
    from qd.qd cimport dd_real, qd_real
    from fplll cimport FT_DD, FT_QD


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
    if float_type_ == FT_LONG_DOUBLE:
        return FP_NR[longdouble].getprec()
    if float_type_ == FT_DPE:
        return FP_NR[dpe_t].getprec()
    IF HAVE_QD:
        if float_type_ == FT_DD:
            return FP_NR[dd_real].getprec()
        if float_type_ == FT_QD:
            return FP_NR[qd_real].getprec()
    if float_type_ == FT_MPFR:
        return FP_NR[mpfr_t].getprec()
    raise ValueError("Floating point type '%s' unknown."%float_type)

def set_precision(unsigned int prec):
    """Set precision globally for MPFR

    :param prec: an integer >= 53
    :returns: current precision

    """
    if prec < 53:
        raise ValueError("Precision (%d) too small."%prec)
    return FP_NR[mpfr_t].setprec(prec)

# TODO fix the interface
def compute_gaussian_heuristic(int block_size, double root_det, double gh_factor):
    """
    Use Gaussian Heuristic to compute a bound on the length of the shortest vector.

    :param int block_size: block size
    :param root_det: root of the determinant
    :param double gh_factor: Gaussian heuristic factor to use

    :returns: (max_dist, max_dist_expo)

    ..  note:: we call ``compute_gaussian_heuristic`` which is declared in bkz.h
    """

    cdef FP_NR[double] gh_dist = 0.0
    cdef FP_NR[double] root_det_ = root_det
    cdef int gh_dist_expo = 0
    gaussian_heuristic_c[FP_NR[double]](gh_dist, gh_dist_expo, block_size, root_det_, gh_factor)
    return gh_dist.get_d(), gh_dist_expo


def get_root_det(MatGSO M, int start, int end):
    if M._type == mpz_double:
        return M.core.mpz_double.get_root_det_c(start, end).get_d()
    elif M._type == mpz_ld:
        return M.core.mpz_ld.get_root_det_c(start, end).get_d()
    elif M._type == mpz_dpe:
        return M.core.mpz_dpe.get_root_det_c(start, end).get_d()
    elif M._type == mpz_mpfr:
        return M.core.mpz_mpfr.get_root_det_c(start, end).get_d()
    else:
        IF HAVE_QD:
            if M._type == mpz_dd:
                return M.core.mpz_dd.get_root_det_c(start, end).get_d()
            elif M._type == mpz_qd:
                return M.core.mpz_qd.get_root_det_c(start, end).get_d()
    raise RuntimeError("MatGSO object '%s' has no core."%M)

class ReductionError(RuntimeError):
    pass
