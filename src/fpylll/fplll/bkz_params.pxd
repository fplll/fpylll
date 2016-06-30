# -*- coding: utf-8 -*-

from decl cimport bkz_auto_abort_core_t, fplll_type_t
from gso cimport MatGSO
from fplll cimport BKZParam as BKZParam_c
from fplll cimport Pruning as Pruning_c
from fplll cimport Strategy as Strategy_c

cdef class Pruning:
    cdef Pruning_c _core
    cdef readonly float radius_factor
    cdef readonly tuple coefficients
    cdef readonly float probability

    @staticmethod
    cdef Pruning from_cxx(Pruning_c & p)

    @staticmethod
    cdef to_cxx(Pruning_c& self, Pruning p)


cdef class Strategy:
    cdef Strategy_c _core
    cdef readonly tuple pruning_parameters
    cdef readonly tuple preprocessing_blocksizes

    @staticmethod
    cdef Strategy from_cxx(Strategy_c & s)

    @staticmethod
    cdef to_cxx(Strategy_c& self, Strategy s)


cdef class BKZParam:
    cdef BKZParam_c *o
    cdef BKZParam _preprocessing

cdef class BKZAutoAbort:
    cdef fplll_type_t _type
    cdef bkz_auto_abort_core_t _core

    cdef MatGSO M
