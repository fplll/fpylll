# -*- coding: utf-8 -*-
"""
Shortest and Closest Vectors.

.. moduleauthor:: Martin R. Albrecht <martinralbrecht+fpylll@googlemail.com>
"""

include "fpylll/config.pxi"

import warnings
from cysignals.signals cimport sig_on, sig_off

from libcpp.vector cimport vector
from fpylll.gmp.mpz cimport mpz_t
from .fplll cimport Z_NR, ZT_MPZ
from .fplll cimport SVP_DEFAULT, CVP_DEFAULT
from .fplll cimport SVP_VERBOSE, CVP_VERBOSE
from .fplll cimport SVP_OVERRIDE_BND
from .fplll cimport SVPM_PROVED, SVPM_FAST
from .fplll cimport SVPMethod
from .fplll cimport CVPM_PROVED, CVPM_FAST
from .fplll cimport CVPMethod
from .fplll cimport shortest_vector_pruning
from .fplll cimport shortest_vector as shortest_vector_c
from .fplll cimport closest_vector as closest_vector_c
from .fplll cimport vector_matrix_product
from .fplll cimport FPLLL_MAX_ENUM_DIM as MAX_ENUM_DIM
from .gso import GSO
from .lll import LLL
from .bkz import BKZ
from .bkz_param import load_strategies_json
from fpylll.algorithms.bkz2 import BKZReduction
from .pruner import Pruning
from fpylll.io cimport assign_Z_NR_mpz, mpz_get_python
from fpylll.io import SuppressStream
from fpylll.util import ReductionError
from fpylll.algorithms.babai import babai

from .integer_matrix cimport IntegerMatrix

def shortest_vector(IntegerMatrix B, method="fast", int flags=SVP_DEFAULT, pruning=True, preprocess=True, max_aux_solutions=0):
    """Return a shortest vector. The result is guaranteed if ``method`` is "proved".

    :param B: A lattice basis
    :param method: One of "fast" or "proved".
    :param int flags:
    :param pruning: If ``True`` pruning parameters are computed by this function.
    :param preprocess: Blocksize used for preprocessing; if  ``True`` a block size is picked.
    :param max_aux_solutions: maximum number of additional short-ish solutions to return

    """

    d = B.nrows

    if pruning is True and d <= 20:
        pruning = None # HACK: pruning in small dimensions can go wrong.

    if d > MAX_ENUM_DIM:
        raise NotImplementedError("This build of FPLLL is configured with a maximum enumeration dimension of %d."%MAX_ENUM_DIM)

    if B._type != ZT_MPZ:
        raise NotImplementedError("Only integer matrices over GMP integers (mpz_t) are supported.")

    cdef SVPMethod method_

    if method == "proved":
        method_ = SVPM_PROVED
        if pruning is True:
            pruning = None
        if pruning is not None:
            raise ValueError("Method 'proved' is incompatible with providing pruning parameters.")
    elif method == "fast":
        method_ = SVPM_FAST
    else:
        raise ValueError("Method '{}' unknown".format(method))

    cdef int r = 0
    s = load_strategies_json(BKZ.DEFAULT_STRATEGY)[-1]

    if preprocess is True and d > s.block_size:
        preprocess = max(min(d-10, s.block_size), 2)

    if preprocess == 2: # just run LLL
        B = LLL.reduction(B)
    elif preprocess is True: # automatic choice
        bkz_obj = BKZReduction(B)
        bkz_obj.svp_reduction(0, d, BKZ.EasyParam(d))
    elif preprocess and preprocess > 2: # make something work
        preprocess = max(min(d-10, preprocess), 2)
        bkz_obj = BKZReduction(B)
        bkz_obj(BKZ.EasyParam(preprocess))

    if pruning is True:
        M = GSO.Mat(B)
        M.update_gso()
        for cost in (10, 20, 30, 40, 50):
            try:
                with SuppressStream():
                    pruning = Pruning.run(M.get_r(0, 0), 2**cost, M.r(), 0.99, flags=Pruning.SINGLE|Pruning.GRADIENT)
                pruning = pruning.coefficients
                break
            except RuntimeError:
                pass

    if pruning is True: # it didn't work
        warnings.warn("Pruning failed, proceeding without it.", RuntimeWarning)
        pruning = [1]*d

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

        if max_aux_solutions == 0:
            sig_on()
            r = shortest_vector_pruning(B._core.mpz[0], sol_coord, pruning_, flags)
            sig_off()
        else:
            sig_on()
            r = shortest_vector_pruning(B._core.mpz[0], sol_coord, auxsol_coord, auxsol_dist, max_aux_solutions, pruning_, flags)
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
    if max_aux_solutions > 0:
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

def closest_vector(IntegerMatrix B, t, method="fast", int flags=CVP_DEFAULT):
    """Return a closest vector.

    The basis must be LLL-reduced with delta=``LLL.DEFAULT_DELTA`` and eta=``LLL.DEFAULT_ETA``.  The
    result is guaranteed if method = "proved", default is "fast".

    :param IntegerMatrix B: Input lattice basis.
    :param t: Target point (∈ ZZ^n)
    :param method: One of "fast" or "proved".
    :param int flags: Either ``CVP.DEFAULT`` or ``CVP.VERBOSE``.
    :returns coordinates of the solution vector:

    EXAMPLE::

        >>> from fpylll import *
        >>> FPLLL.set_random_seed(42)
        >>> A = IntegerMatrix.random(5, 'uniform', bits=10)
        >>> lll = LLL.reduction(A)
        >>> t = (94, -42, 123, 512, -1337)
        >>> print (CVP.closest_vector(A, t))
        (-34, 109, 204, 360, -1548)

       >>> from fpylll import *
       >>> n = 10
       >>> B = IntegerMatrix(n, n + 1)
       >>> B.randomize("intrel", bits=100)
       >>> v_opt = B.multiply_left([1,0,1,0,1,1,0,0,1,1])
       >>> s = v_opt[0] # s = <a, x>, where a is vector of knapsack values.
       >>> t = [s] + (n * [0])
       >>> _ = LLL.reduction(B)
       >>> v = CVP.closest_vector(B, t)
       >>> v[0] == t[0]
       True
       >>> v[1:]
       (1, 0, 1, 0, 1, 1, 0, 0, 1, 1)

    """

    if B.nrows > MAX_ENUM_DIM:
        raise NotImplementedError("This build of FPLLL is configured with a maximum enumeration dimension of %d."%MAX_ENUM_DIM)

    if B._type != ZT_MPZ:
        raise NotImplementedError("Only integer matrices over GMP integers (mpz_t) are supported.")

    cdef CVPMethod method_
    if method == "proved":
        method_ = CVPM_PROVED
    elif method == "fast":
        method_ = CVPM_FAST
    else:
        raise ValueError("Method '{}' unknown".format(method))

    cdef int r = 0

    cdef vector[Z_NR[mpz_t]] int_target
    cdef vector[Z_NR[mpz_t]] sol_coord
    cdef vector[Z_NR[mpz_t]] solution

    int_target.resize(len(t))

    for i in range(len(t)):
        assign_Z_NR_mpz(int_target[i], t[i])

    sig_on()
    r = closest_vector_c(B._core.mpz[0], int_target, sol_coord, method_, flags)
    sig_off()

    if r:
        raise ReductionError("CVP solver returned an error ({:d})".format(r))

    vector_matrix_product(solution, sol_coord, B._core.mpz[0])

    cdef list v = []

    for i in range(solution.size()):
        v.append(mpz_get_python(solution[i].get_data()))

    return tuple(v)

class CVP:
    babai = staticmethod(babai)
    closest_vector = staticmethod(closest_vector)
    DEFAULT = CVP_DEFAULT
    VERBOSE = CVP_VERBOSE
