# -*- coding: utf-8 -*-
"""
Shortest and Closest Vectors.

.. moduleauthor:: Martin R. Albrecht <martinralbrecht+fpylll@googlemail.com>
"""

include "fpylll/config.pxi"

import threading
from cysignals.signals cimport sig_on, sig_off

from libcpp.vector cimport vector
from fpylll.gmp.mpz cimport mpz_t
from .fplll cimport Z_NR, ZT_MPZ
from .fplll cimport SVP_DEFAULT, CVP_DEFAULT
from .fplll cimport SVP_VERBOSE, CVP_VERBOSE
from .fplll cimport SVP_OVERRIDE_BND
from .fplll cimport SVPM_PROVED, SVPM_FAST
from .fplll cimport SVPMethod
from .fplll cimport shortest_vector_pruning
from .fplll cimport shortest_vector as shortest_vector_c
from .fplll cimport closest_vector as closest_vector_c
from .fplll cimport vector_matrix_product
from .lll import lll_reduction
from fpylll.io cimport assign_Z_NR_mpz, mpz_get_python
from fpylll.util import ReductionError

from .integer_matrix cimport IntegerMatrix

def shortest_vector(IntegerMatrix B, method=None, int flags=SVP_DEFAULT, pruning=None, run_lll=True, max_aux_sols=0):
    """Return a shortest vector.

    :param IntegerMatrix B:
    :param method:
    :param int flags:
    :param pruning:
    :param run_lll:
    :param max_aux_sols:
    :returns:
    :rtype:

    """

    if B._type != ZT_MPZ:
        raise NotImplementedError("Only integer matrices over GMP integers (mpz_t) are supported.")

    cdef SVPMethod method_
    if method == "proved" or method is None:
        method_ = SVPM_PROVED
    elif method == "fast":
        method_ = SVPM_FAST
    else:
        raise ValueError("Method '{}' unknown".format(method))

    cdef int r = 0

    if run_lll:
        lll_reduction(B)

    cdef vector[Z_NR[mpz_t]] sol_coord
    cdef vector[Z_NR[mpz_t]] solution
    cdef vector[double] pruning_

    cdef vector[vector[Z_NR[mpz_t]]] auxsol_coord
    cdef vector[double] auxsol_dist

    if pruning:
        if len(pruning) != B.nrows:
            raise ValueError("PruningParams vector must have length %d but got %d."%(B.nrows, len(pruning)))

        pruning_.resize(B.nrows)
        for i in range(len(pruning)):
            pruning_[i] = pruning[i]

        if max_aux_sols == 0:
            sig_on()
            r = shortest_vector_pruning(B._core.mpz[0], sol_coord, pruning_, flags)
            sig_off()
        else:
            sig_on()
            r = shortest_vector_pruning(B._core.mpz[0], sol_coord, auxsol_coord, auxsol_dist, max_aux_sols, pruning_, flags)
            sig_off()
    else:
        sig_on()
        r = shortest_vector_c(B._core.mpz[0], sol_coord, method_, flags)
        sig_off()

    if r:
        raise ReductionError("SVP solver returned an error ({:d})".format(r))

    vector_matrix_product(solution, sol_coord, B._core.mpz[0])

    cdef list v = []

    for i in range(solution.size()):
        v.append(mpz_get_python(solution[i].get_data()))

    cdef list aux = []
    if max_aux_sols > 0:
        for j in range(auxsol_dist.size()):
            vector_matrix_product(solution, auxsol_coord[j], B._core.mpz[0])
            aux_sol = []
            for i in range(solution.size()):
                aux_sol.append(mpz_get_python(solution[i].get_data()))
            aux.append(tuple(aux_sol))
        return tuple(v), tuple(aux)
    else:
        return tuple(v)

class SVP:
    shortest_vector = staticmethod(shortest_vector)
    DEFAULT = SVP_DEFAULT
    VERBOSE = SVP_VERBOSE
    OVERRIDE_BND = SVP_OVERRIDE_BND

def closest_vector(IntegerMatrix B, target, int flags=CVP_DEFAULT):
    """Return a closest vector.

    :param IntegerMatrix B:
    :param vector[Z_NR[mpz_t]] target:
    :param int flags:
    :returns coordinates of the solution vector:
    :rtype tuple:

    >>> from fpylll import *
    >>> FPLLL.set_random_seed(42)
    >>> A = IntegerMatrix.random(5, 'uniform', bits=10)
    >>> lll = LLL.reduction(A)
    >>> t = (94, -42, 123, 512, -1337)
    >>> print (CVP.closest_vector(A, t))
    (-34, 109, 204, 360, -1548)

    """

    if B._type != ZT_MPZ:
        raise NotImplementedError("Only integer matrices over GMP integers (mpz_t) are supported.")

    cdef int r = 0

    cdef vector[Z_NR[mpz_t]] int_target
    cdef vector[Z_NR[mpz_t]] sol_coord
    cdef vector[Z_NR[mpz_t]] solution

    int_target.resize(len(target))
    cdef Z_NR[mpz_t] t

    for i in range(len(target)):
        assign_Z_NR_mpz(int_target[i], target[i])

    sig_on()
    r = closest_vector_c(B._core.mpz[0], int_target, sol_coord, flags)
    sig_off()

    if r:
        raise ReductionError("CVP solver returned an error ({:d})".format(r))

    vector_matrix_product(solution, sol_coord, B._core.mpz[0])

    cdef list v = []

    for i in range(solution.size()):
        v.append(mpz_get_python(solution[i].get_data()))

    return tuple(v)

class CVP:
    closest_vector = staticmethod(closest_vector)
    DEFAULT = CVP_DEFAULT
    VERBOSE = CVP_VERBOSE
