# -*- coding: utf-8 -*-

from .decl cimport enumeration_core_t, evaluator_core_t, fplll_mat_gso_data_type_t
from .fplll cimport PyCallbackEvaluatorWrapper as PyCallbackEvaluatorWrapper_c
from .gso cimport MatGSO

cdef class Enumeration:
    cdef readonly MatGSO M
    cdef enumeration_core_t _core
    cdef evaluator_core_t _eval_core
    cdef PyCallbackEvaluatorWrapper_c *_callback_wrapper
