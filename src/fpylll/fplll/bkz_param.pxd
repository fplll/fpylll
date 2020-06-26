# -*- coding: utf-8 -*-

from libcpp.vector cimport vector
from .decl cimport bkz_auto_abort_core_t, fplll_mat_gso_data_type_t
from .gso cimport MatGSO
from .fplll cimport BKZParam as BKZParam_c
from .fplll cimport PruningParams as PruningParams_c
from .fplll cimport Strategy as Strategy_c
from .fplll cimport PrunerMetric

cdef class Strategy:
    cdef Strategy_c _core
    cdef dict aux

    @staticmethod
    cdef Strategy from_cxx(Strategy_c & s)

    @staticmethod
    cdef to_cxx(Strategy_c& self, Strategy s)


cdef class BKZParam:
    # BKZParam_c doesn't actually store strategies, store them here
    cdef vector[Strategy_c] strategies_c
    cdef BKZParam_c *o
    cdef readonly tuple strategies
    cdef dict aux
