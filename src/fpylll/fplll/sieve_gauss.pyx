# -*- coding: utf-8 -*-
"""
Gaussian sieving.

..  moduleauthor:: Martin R.  Albrecht <martinralbrecht+fpylll@googlemail.com>

Gaussian Sieving was proposed in Micciancio, D., & Voulgaris, P.  (2010).  Faster exponential time
algorithms for the shortest vector problem.  In M.  Charika, 21st SODA (pp.  1468–1480).  :
ACM-SIAM.

Example::

    >>> from fpylll import IntegerMatrix, GaussSieve, SVP, LLL, FPLLL
    >>> FPLLL.set_random_seed(1337)
    >>> A = IntegerMatrix.random(20, "qary", k=10, q=127); A = LLL.reduction(A)
    >>> w = SVP.shortest_vector(A)
    >>> v = GaussSieve(A, algorithm=2)()
    >>> sum([w_**2 for w_ in w]) == sum([v_**2 for v_ in v])
    True

"""

include "fpylll/config.pxi"

from random import randint
from .fplll cimport NumVect, Z_NR, ZT_MPZ, ZT_LONG
from .fplll cimport GaussSieve as GaussSieve_c
from fpylll.io cimport assign_Z_NR_mpz, mpz_get_python
from .integer_matrix cimport IntegerMatrix
from cysignals.signals cimport sig_on, sig_off


cdef class GaussSieve:
    def __init__(self, IntegerMatrix A, int algorithm, int verbose=0, seed=None):
        """Create new Gaussian Sieve

        :param IntegerMatrix A: sieving will be performed over the whole basis
        :param algorithm: one of 2,3 or 4 for 2-, 3- or 4-sieving respectively

        """
        if seed is None:
            seed = randint(0, 2**31)

        if algorithm not in (2,3,4):
            raise ValueError("Algorithm must be one of 2, 3 or 4, but received %d"%algorithm)

        if A._type == ZT_MPZ:
            self._core.mpz_d = new GaussSieve_c[mpz_t, FP_NR[double]](A._core.mpz[0], algorithm, verbose, seed)
        elif A._type == ZT_LONG:
            self._core.long_d = new GaussSieve_c[long, FP_NR[double]](A._core.long[0], algorithm, verbose, seed)
        else:
            raise RuntimeError("Integer type '%s' not understood."%A._type)

        self._type = A._type

    def __dealloc__(self):
        """
        Delete sieve
        """
        if self._type == ZT_MPZ:
            del self._core.mpz_d
        elif self._type == ZT_LONG:
            del self._core.long_d

    def __call__(self, int target_norm=0):
        """
        Call sieving algorithm and return shortest vector found

        :param target_norm:

        """

        cdef Z_NR[mpz_t] target_norm_mpz_
        cdef Z_NR[long] target_norm_l_
        cdef NumVect[Z_NR[mpz_t]] r_mpz_
        cdef NumVect[Z_NR[long]] r_l_
        cdef list r  = []

        if self._type == ZT_MPZ:
            assign_Z_NR_mpz(target_norm_mpz_, target_norm)
            sig_on()
            self._core.mpz_d.sieve(target_norm_mpz_)
            sig_off()

            r_mpz_ = self._core.mpz_d.return_first()

            for i in range(r_mpz_.size()):
                r.append(mpz_get_python(r_mpz_[i].get_data()))
        elif self._type == ZT_LONG:
            target_norm_l_ = target_norm
            sig_on()
            self._core.long_d.sieve(target_norm_l_)
            sig_off()

            r_l_ = self._core.long_d.return_first()

            for i in range(r_l_.size()):
                r.append(r_l_[i].get_data())
        else:
            RuntimeError("Integer type '%s' not understood."%self._type)

        return tuple(r)

    @property
    def verbose(self):
        """
        >>> from fpylll import IntegerMatrix, GaussSieve, SVP, LLL
        >>> A = IntegerMatrix.random(30, "qary", k=15, q=127); A = LLL.reduction(A)
        >>> GaussSieve(A, 2, verbose=True).verbose
        True

        """
        if self._type == ZT_MPZ:
            return self._core.mpz_d.verbose
        elif self._type == ZT_LONG:
            return self._core.long_d.verbose
        else:
            RuntimeError("Integer type '%s' not understood."%self._type)

    @verbose.setter
    def verbose(self, value):
        """
        >>> from fpylll import IntegerMatrix, GaussSieve, SVP, LLL
        >>> A = IntegerMatrix.random(30, "qary", k=15, q=127); A = LLL.reduction(A)
        >>> sieve = GaussSieve(A, 2, verbose=True)
        >>> sieve.verbose = False
        >>> sieve.verbose
        False

        """

        if self._type == ZT_MPZ:
            self._core.mpz_d.set_verbose(value)
        elif self._type == ZT_LONG:
            self._core.long_d.set_verbose(value)
        else:
            RuntimeError("Integer type '%s' not understood."%self._type)
