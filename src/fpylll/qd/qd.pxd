# -*- coding: utf-8 -*-


cdef extern from "qd/dd_real.h":
    cdef cppclass dd_real:
        dd_real(double hi, double lo)
        dd_real()

cdef extern from "qd/qd_real.h":
    cdef cppclass qd_real:
        qd_real(double hi, double lo)
        qd_real()
