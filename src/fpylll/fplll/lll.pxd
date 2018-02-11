# -*- coding: utf-8 -*-

from .gso cimport MatGSO
from .decl cimport lll_reduction_core_t, fplll_mat_gso_data_type_t

cdef class LLLReduction:

    cdef fplll_mat_gso_data_type_t _type
    cdef lll_reduction_core_t _core

    cdef readonly MatGSO M
    cdef double _delta
    cdef double _eta
