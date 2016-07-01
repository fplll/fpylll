# -*- coding: utf-8 -*-
include "fpylll/config.pxi"
include "cysignals/signals.pxi"


from fplll cimport Matrix, Z_NR, mpz_t
from fplll cimport LLL_DEF_ETA, LLL_DEF_DELTA, LLL_DEFAULT
from fplll cimport get_red_status_str
from fpylll.util import ReductionError


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
        self.B = B
        # TODO: Don't hardcode this
        self.U = IntegerMatrix(0,0)
        self.UinvT = IntegerMatrix(0,0)

        self._core = new Wrapper_c((self.B._core)[0],
                                   (self.U._core)[0],
                                   (self.UinvT._core)[0],
                                   delta, eta, flags)
        self._called = False

    def __dealloc__(self):
        del self._core

    def __reduce__(self):
        """
        Make sure attempts at pickling raise an error until proper pickling is implemented.
        """
        raise NotImplementedError

    def __call__(self):
        """Run LLL.

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

