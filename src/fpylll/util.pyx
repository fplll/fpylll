from fplll cimport FT_DEFAULT, FT_DOUBLE, FT_LONG_DOUBLE, FT_DD, FT_QD, FT_DPE, FT_MPFR
from fplll cimport FP_NR, RandGen
from cpython.int cimport PyInt_AS_LONG
from fpylll.gmp.pylong cimport mpz_get_pyintlong, mpz_set_pylong
from fpylll.gmp.mpz cimport mpz_init, mpz_clear, mpz_set_si
from fpylll.gmp.random cimport gmp_randstate_t, gmp_randseed_ui
from fpylll.mpfr.mpfr cimport mpfr_t

cdef FloatType check_float_type(object float_type):
    cdef FloatType float_type_

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

cdef int assign_Z_NR_mpz(Z_NR[mpz_t]& t, value) except -1:
    cdef mpz_t tmp
    mpz_init(tmp)
    if isinstance(value, int):
        mpz_set_si(tmp, PyInt_AS_LONG(value))
    elif isinstance(value, long):
        mpz_set_pylong(tmp, value)
    else:
        mpz_clear(tmp)
        msg = "Only Python ints and longs are currently supported, but got type '%s'"%type(value)
        raise NotImplementedError(msg)

    t.set(tmp)
    mpz_clear(tmp)


def set_random_seed(unsigned long seed):
    """Set random seed.

    :param seed: a new seed.

    """
    cdef gmp_randstate_t state = RandGen.getGMPState()
    gmp_randseed_ui(state, seed)

def get_precision():
    """Get currently set precision for MPFR

    :returns: precision in bits

    """
    return FP_NR[mpfr_t].getprec()

def set_precision(unsigned int prec):
    """Set precision globally for MPFR

    :param prec: an integer >= 53
    :returns: current precision

    """
    if prec < 53:
        raise ValueError("Precision (%d) too small."%prec)
    return FP_NR[mpfr_t].setprec(prec)
