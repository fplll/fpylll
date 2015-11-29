# -*- coding: utf-8 -*-
# distutils: language = c++
# distutils: libraries = gmp mpfr fplll
"""
.. moduleauthor:: Martin R. Albrecht <martinralbrecht+fpylll@googlemail.com>
"""

include "interrupt/interrupt.pxi"

from libcpp.vector cimport vector
from gmp.mpz cimport mpz_t
from gmp.pylong cimport mpz_get_pyintlong
from fplll cimport Z_NR
from fplll cimport SVP_DEFAULT, CVP_DEFAULT
from fplll cimport SVP_VERBOSE, CVP_VERBOSE
from fplll cimport SVP_OVERRIDE_BND
from fplll cimport SVPM_PROVED, SVPM_FAST
from fplll cimport SVPMethod
from fplll cimport shortestVectorPruning, shortestVector
from fplll cimport vectMatrixProduct
from lll import lll_reduction
from util cimport assign_Z_NR_mpz
from fpylll import ReductionError

from integer_matrix cimport IntegerMatrix

def shortest_vector(IntegerMatrix B, method=None, int flags=SVP_DEFAULT, max_dist=None, pruning=None, run_lll=True):
    """Return a shortest vector.

    :param IntegerMatrix B:
    :param method:
    :param int flags:
    :param max_dist:
    :param pruning:
    :returns:
    :rtype:

    """
    cdef SVPMethod method_
    if method == "proved" or method is None:
        method_ = SVPM_PROVED
    elif method == "fast":
        method_ = SVPM_FAST
    else:
        raise ValueError("Method '{}' unknown".format(method))

    cdef int r

    if run_lll:
        lll_reduction(B)

    cdef vector[Z_NR[mpz_t]] solCoord
    cdef vector[Z_NR[mpz_t]] solution
    cdef vector[double] pruning_
    cdef Z_NR[mpz_t] max_dist_

    if pruning:
        if len(pruning) != B.nrows:
            raise ValueError("Pruning vector must have length %d but got %d."%(B.nrows, len(pruning)))

        pruning_.resize(B.nrows)
        for i in range(len(pruning)):
            pruning_[i] = pruning[i]

        if max_dist:
            assign_Z_NR_mpz(max_dist_, max_dist)

        sig_on()
        r = shortestVectorPruning(B._core[0], solCoord, pruning_, max_dist_, flags)
        sig_off()
    else:
        sig_on()
        r = shortestVector(B._core[0], solCoord, method_, flags)
        sig_off()


    if r:
        raise ReductionError("SVP solver returned an error ({:d})".format(r))

    vectMatrixProduct(solution, solCoord, B._core[0])

    cdef list v = []

    for i in range(solution.size()):
        v.append(mpz_get_pyintlong(solution[i].getData()))

    return tuple(v)

class SVP:
    shortest_vector = shortest_vector
    DEFAULT = SVP_DEFAULT
    VERBOSE = SVP_VERBOSE
    OVERRIDE_BND = SVP_OVERRIDE_BND

# class CVP:
#     closest_vector = closest_vector
#     DEFAULT = CVP_DEFAULT
#     VERBOSE = CVP_VERBOSE
