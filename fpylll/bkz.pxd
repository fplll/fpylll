# -*- coding: utf-8 -*-
# distutils: language = c++
# distutils: libraries = gmp mpfr fplll

from fpylll cimport bkz_auto_abort_core, fplll_type
from gso cimport MatGSO

cdef class BKZAutoAbort:
    cdef fplll_type _type
    cdef bkz_auto_abort_core _core

    cdef MatGSO m
