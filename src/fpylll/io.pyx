# -*- coding: utf-8 -*-
include "fpylll/config.pxi"

import sys
import os

from cpython.int cimport PyInt_AS_LONG
from fpylll.gmp.mpz cimport mpz_init, mpz_clear, mpz_set
from fpylll.gmp.pylong cimport mpz_get_pyintlong, mpz_set_pylong
from .gmp.mpz cimport mpz_t, mpz_set_si, mpz_set
from cpython.version cimport PY_MAJOR_VERSION
from fplll.fplll cimport FT_DEFAULT, FT_DOUBLE, FT_LONG_DOUBLE, FT_DPE, FT_MPFR
from fplll.fplll cimport ZT_MPZ, ZT_LONG

# Note: this uses fpylll's numpy and not the global numpy package.
IF HAVE_NUMPY:
    from .numpy import is_numpy_integer

IF HAVE_QD:
    from fpylll.fplll.fplll cimport FT_DD, FT_QD

try:
    from sage.rings.integer import Integer
    have_sage = True
except Exception:
    have_sage = False

cdef int assign_Z_NR_mpz(Z_NR[mpz_t]& t, value) except -1:
    """
    Assign Python integer to Z_NR[mpz_t]
    """
    cdef mpz_t tmp
    mpz_init(tmp)
    try:
        assign_mpz(tmp, value)
        t.set(tmp)
    finally:
        mpz_clear(tmp)

cdef int assign_mpz(mpz_t& t, value) except -1:
    """
    Assign Python integer to Z_NR[mpz_t]
    """     
    if isinstance(value, int) and PY_MAJOR_VERSION == 2:
            mpz_set_si(t, PyInt_AS_LONG(value))
            return 0
    if isinstance(value, long):
        mpz_set_pylong(t, value)
        return 0
    if have_sage:
        if isinstance(value, Integer):
            value = long(value)
            mpz_set_pylong(t, value)
            return 0

    IF HAVE_NUMPY:
        if is_numpy_integer(value):
            value = long(value)
            mpz_set_pylong(t, value)
            return 0

    raise NotImplementedError("Type '%s' not supported"%type(value))

cdef object mpz_get_python(mpz_srcptr z):
    r = mpz_get_pyintlong(z)
    if have_sage:
        return Integer(r)
    else:
        return r

cdef void vector_fp_nr_barf(vector_fp_nr_t &out, object inp, FloatType float_type):
    cdef fp_nr_t tmp
    cdef bytes py_bytes
    if float_type == FT_DOUBLE:
        for entry in inp:
            # this is slow but we want to cover all kinds of Python types here
            py_bytes = str(entry).encode()
            tmp.d = <char*>py_bytes
            out.d.push_back(tmp.d)
    elif float_type == FT_LONG_DOUBLE:
        IF HAVE_LONG_DOUBLE:
            for entry in inp:
                py_bytes = str(entry).encode()
                tmp.ld = <char*>py_bytes
                out.ld.push_back(tmp.ld)
        ELSE:
            raise ValueError("Float type '%s' not understood."%float_type)
    elif float_type == FT_DPE:
        for entry in inp:
            py_bytes = str(entry).encode()
            tmp.dpe = <char*>py_bytes
            out.dpe.push_back(tmp.dpe)
    elif float_type == FT_MPFR:
        for entry in inp:
            py_bytes = str(entry).encode()
            tmp.mpfr = <char*>py_bytes
            out.mpfr.push_back(tmp.mpfr)
    else:
        IF HAVE_QD:
            if float_type == FT_DD:
                for entry in inp:
                    py_bytes = str(entry).encode()
                    tmp.dd = <char*>py_bytes
                    out.dd.push_back(tmp.dd)
            elif float_type == FT_QD:
                for entry in inp:
                    py_bytes = str(entry).encode()
                    tmp.qd = <char*>py_bytes
                    out.qd.push_back(tmp.qd)
            else:
                raise ValueError("Float type '%s' not understood."%float_type)
        ELSE:
            raise ValueError("Float type '%s' not understood."%float_type)

cdef object vector_fp_nr_slurp(vector_fp_nr_t &inp, FloatType float_type):
    out = []
    if float_type == FT_DOUBLE:
        for i in range(inp.d.size()):
            out.append(inp.d[i].get_d())
    elif float_type == FT_LONG_DOUBLE:
        IF HAVE_LONG_DOUBLE:
            for i in range(inp.ld.size()):
                out.append(inp.ld[i].get_d())
        ELSE:
            raise ValueError("Float type '%s' not understood."%float_type)
    elif float_type == FT_DPE:
        for i in range(inp.dpe.size()):
            out.append(inp.dpe[i].get_d())
    elif float_type == FT_MPFR:
        for i in range(inp.mpfr.size()):
            out.append(inp.mpfr[i].get_d())
    else:
        IF HAVE_QD:
            if float_type == FT_DD:
                for i in range(inp.dd.size()):
                    out.append(inp.dd[i].get_d())
            elif float_type == FT_QD:
                for i in range(inp.qd.size()):
                    out.append(inp.qd[i].get_d())
            else:
                raise ValueError("Float type '%s' not understood."%float_type)
        ELSE:
            raise ValueError("Float type '%s' not understood."%float_type)
    return tuple(out)

cdef object vector_z_nr_slurp(vector_z_nr_t &inp, IntType int_type):
    out = []
    if int_type == ZT_MPZ:
        for i in range(inp.mpz.size()):
            out.append(mpz_get_python(inp.mpz[i].get_data()))
    elif int_type == ZT_LONG:
        for i in range(inp.long.size()):
            out.append(inp.long[i].get_data())
    else:
        raise ValueError("Int type '%s' not understood."%int_type)
    return tuple(out)


class SuppressStream(object):
    """
    Suppress errors (being printed by FPLLL, which are to be expected).
    """

    def __init__(self, stream=sys.stderr):
        try:
            self.orig_stream_fileno = stream.fileno()
            self.skip = False
        except OSError:
            self.skip = True

    def __enter__(self):
        if self.skip:
            return
        self.orig_stream_dup = os.dup(self.orig_stream_fileno)
        self.devnull = open(os.devnull, "w")
        os.dup2(self.devnull.fileno(), self.orig_stream_fileno)

    def __exit__(self, type, value, traceback):
        if self.skip:
            return
        os.close(self.orig_stream_fileno)
        os.dup2(self.orig_stream_dup, self.orig_stream_fileno)
        os.close(self.orig_stream_dup)
        self.devnull.close()
