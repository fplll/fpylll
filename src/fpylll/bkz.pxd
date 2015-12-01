# -*- coding: utf-8 -*-
# distutils: language = c++
# distutils: libraries = gmp mpfr fplll

from fpylll cimport bkz_auto_abort_core, fplll_type_t
from gso cimport MatGSO
from fplll cimport BKZParam as BKZParam_c

cdef class BKZParam:
    cdef BKZParam_c *o
    cdef BKZParam _preprocessing

cdef class BKZAutoAbort:
    cdef fplll_type_t _type
    cdef bkz_auto_abort_core _core

    cdef MatGSO M
