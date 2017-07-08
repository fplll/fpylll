# -*- coding: utf-8 -*-

from integer_matrix cimport IntegerMatrix
from decl cimport pruner_core_t, fplll_nr_type_t

cdef class Pruner:
    cdef fplll_nr_type_t _type
    cdef pruner_core_t _core
