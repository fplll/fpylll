# -*- coding: utf-8 -*-

from .integer_matrix cimport IntegerMatrix
from .decl cimport mat_gso_core_t, fplll_mat_gso_data_type_t, fplll_mat_gso_alg_type_t

cdef class MatGSO:
    cdef fplll_mat_gso_data_type_t _type
    cdef fplll_mat_gso_alg_type_t  _alg
    cdef mat_gso_core_t _core

    cdef readonly IntegerMatrix B
    cdef readonly IntegerMatrix _G
    cdef readonly IntegerMatrix U
    cdef readonly IntegerMatrix UinvT
