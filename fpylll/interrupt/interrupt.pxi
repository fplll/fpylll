from fpylll.interrupt.interrupt cimport *

cdef extern from 'interrupt/pxi.h':
    int import_fpylll__interrupt__interrupt() except -1

# This *must* be done for every module using interrupt functions
# otherwise you will get segmentation faults.
import_fpylll__interrupt__interrupt()
