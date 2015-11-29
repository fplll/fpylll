# -*- coding: utf-8 -*-
# distutils: language = c++
# distutils: libraries = gmp mpfr fplll

include "interrupt/interrupt.pxi"

from fplll cimport Matrix, Z_NR, mpz_t
from fplll cimport LLL_DEF_ETA, LLL_DEF_DELTA, LLL_DEFAULT
from fplll cimport getRedStatusStr
from fpylll import ReductionError


cdef class Wrapper:
    def __init__(self, IntegerMatrix B, double delta=LLL_DEF_DELTA, double eta=LLL_DEF_ETA, int flags=LLL_DEFAULT):
        """FIXME! briefly describe function

        :param IntegerMatrix B:
        :param double delta:
        :param double eta:
        :param int flags:


        >>> from fpylll import LLL, IntegerMatrix
        >>> A = IntegerMatrix(50, 50)
        >>> A.randomize("ntrulike", bits=100, q=1023)
        >>> W = LLL.Wrapper(A)

        """
        self._B = B
        self._U = IntegerMatrix(0,0)
        self._UinvT = IntegerMatrix(0,0)

        self._core = new Wrapper_c((self._B._core)[0],
                                   (self._U._core)[0],
                                   (self._UinvT._core)[0],
                                   delta, eta, flags)
        self._called = False

    def __dealloc__(self):
        del self._core

    def __call__(self):
        """FIXME! briefly describe function

        :returns:
        :rtype:

        >>> from fpylll import LLL, IntegerMatrix, GSO
        >>> A = IntegerMatrix(40, 40)
        >>> A.randomize("ntrulike", bits=10, q=1023)
        >>> W = LLL.Wrapper(A)
        >>> W()

        """

        if self._called:
            raise ValueError("lll() may only be called once.")

        sig_on()
        self._core.lll()
        sig_off()
        self._called = True


    @property
    def status(self):
        return self._core.status

    @property
    def B(self):
        return self._B

    @property
    def U(self):
        return self._U

    @property
    def UinvT(self):
        return self._UinvT
