# -*- coding: utf-8 -*-

from decl cimport bkz_auto_abort_core_t, fplll_type_t
from gso cimport MatGSO

cdef class BKZAutoAbort:
    cdef fplll_type_t _type
    cdef bkz_auto_abort_core_t _core

    cdef MatGSO M
