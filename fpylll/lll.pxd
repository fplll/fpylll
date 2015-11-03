# -*- coding: utf-8 -*-
# distutils: language = c++
# distutils: libraries = gmp mpfr fplll

from gso cimport MatGSO
from fpylll cimport lll_reduction_core, fplll_type

cdef class LLLReduction:

    cdef fplll_type _type
    cdef lll_reduction_core _core

    cdef MatGSO m
    cdef double _delta
    cdef double _eta
