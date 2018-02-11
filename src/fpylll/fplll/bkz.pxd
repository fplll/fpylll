# -*- coding: utf-8 -*-

from .decl cimport bkz_auto_abort_core_t, fplll_mat_gso_data_type_t
from .decl cimport bkz_reduction_core_t
from .gso cimport MatGSO
from .bkz_param cimport BKZParam
from .lll cimport LLLReduction

cdef class BKZAutoAbort:
    cdef fplll_mat_gso_data_type_t _type
    cdef bkz_auto_abort_core_t _core

    cdef MatGSO M


cdef class BKZReduction:

    cdef fplll_mat_gso_data_type_t _type
    cdef bkz_reduction_core_t _core

    cdef readonly MatGSO M
    cdef readonly LLLReduction lll_obj
    cdef readonly BKZParam param

