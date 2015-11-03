from fplll cimport FT_DEFAULT, FT_DOUBLE, FT_LONG_DOUBLE, FT_DPE, FT_MPFR

cdef FloatType check_float_type(object float_type):
    cdef FloatType float_type_

    if float_type == "default" or float_type is None:
        float_type_= FT_DEFAULT
    elif float_type == "double":
        float_type_ = FT_DOUBLE
    elif float_type == "long double":
        float_type_ = FT_LONG_DOUBLE
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

cdef void recursively_free_bkz_param(BKZParam_c *param):
    if not param:
        return
    if param.preprocessing:
        recursively_free_bkz_param(param.preprocessing)
    del param
