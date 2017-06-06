# -*- coding: utf-8 -*-
"""
Gaussian sieving.

..  moduleauthor:: Martin R.  Albrecht <martinralbrecht+fpylll@googlemail.com>

Gaussian Sieving was proposed in Micciancio, D., & Voulgaris, P.  (2010).  Faster exponential time
algorithms for the shortest vector problem.  In M.  Charika, 21st SODA (pp.  1468–1480).  :
ACM-SIAM.

>>> from fpylll import IntegerMatrix, GaussSieve, SVP, LLL
>>> A = IntegerMatrix.random(30, "qary", k=15, q=127); A = LLL.reduction(A)
>>> w = SVP.shortest_vector(A)
>>> v = GaussSieve(A, algorithm=2)()
>>> sum([w_**2 for w_ in w]) == sum([v_**2 for v_ in v])
True

"""

include "fpylll/config.pxi"

from random import randint
from fplll cimport NumVect, Z_NR
from fpylll.io cimport assign_Z_NR_mpz, mpz_get_python
from integer_matrix cimport IntegerMatrix
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

        self._core = new GaussSieve_c[mpz_t, FP_NR[double]](A._core[0], algorithm, verbose, seed)

    def __dealloc__(self):
        """
        Delete sieve
        """
        del self._core

    def __call__(self, int target_norm=0):
        """
        Call sieving algorithm and return shortest vector found

        :param target_norm:

        """

        cdef Z_NR[mpz_t] target_norm_
        assign_Z_NR_mpz(target_norm_, target_norm)

        sig_on()
        self._core.sieve(target_norm_)
        sig_off()

        cdef NumVect[Z_NR[mpz_t]] r_ = self._core.return_first()
        cdef list r  = []

        for i in range(r_.size()):
            r.append(mpz_get_python(r_[i].get_data()))

        return tuple(r)

    @property
    def verbose(self):
        """
        >>> from fpylll import IntegerMatrix, GaussSieve, SVP, LLL
        >>> A = IntegerMatrix.random(30, "qary", k=15, q=127); A = LLL.reduction(A)
        >>> GaussSieve(A, 2, verbose=True).verbose
        True

        """
        return self._core.verbose

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

        self._core.set_verbose(value)
