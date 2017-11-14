# -*- coding: utf-8 -*-

from .integer_matrix cimport IntegerMatrix
from .decl cimport pruner_core_t, fplll_nr_type_t
from .fplll cimport PruningParams as PruningParams_c

cdef class PruningParams:
    cdef PruningParams_c _core

    @staticmethod
    cdef PruningParams from_cxx(PruningParams_c & p)

    @staticmethod
    cdef to_cxx(PruningParams_c& self, PruningParams p)

cdef class Pruner:
    cdef fplll_nr_type_t _type
    cdef pruner_core_t _core
