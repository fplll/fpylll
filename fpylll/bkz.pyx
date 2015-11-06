# -*- coding: utf-8 -*-
# distutils: language = c++
# distutils: libraries = gmp mpfr fplll

from gmp.mpz cimport mpz_t
from mpfr.mpfr cimport mpfr_t

from fplll cimport LLL_DEF_DELTA, FloatType
from fplll cimport BKZParam as BKZParam_c
from fplll cimport BKZAutoAbort as BKZAutoAbort_c
from fplll cimport BKZ_VERBOSE, BKZ_NO_LLL, BKZ_BOUNDED_LLL, BKZ_GH_BND, BKZ_AUTO_ABORT
from fplll cimport BKZ_MAX_LOOPS, BKZ_MAX_TIME, BKZ_DUMP_GSO, BKZ_DEFAULT
from fplll cimport RED_BKZ_LOOPS_LIMIT, RED_BKZ_TIME_LIMIT
from fplll cimport bkzReduction, getRedStatusStr
from fplll cimport FP_NR, Z_NR

from fpylll import ReductionError
from fpylll cimport mpz_double, mpz_mpfr

from integer_matrix cimport IntegerMatrix
from util cimport check_delta, check_precision, check_float_type, recursively_free_bkz_param

include "interrupt/interrupt.pxi"

class BKZFlags:
    DEFAULT = BKZ_DEFAULT
    VERBOSE = BKZ_VERBOSE
    NO_LLL = BKZ_NO_LLL
    BOUNDED_LLL = BKZ_BOUNDED_LLL,
    GH_BND = BKZ_GH_BND
    AUTO_ABORT = BKZ_AUTO_ABORT
    MAX_LOOPS = BKZ_MAX_LOOPS
    MAX_TIME = BKZ_MAX_TIME
    DUMP_GSO = BKZ_DUMP_GSO

cdef class BKZParam:
    def __init__(self, int block_size, float delta=LLL_DEF_DELTA, int flags=BKZ_DEFAULT,
                 int max_loops=0, int max_time=0,
                 auto_abort=None, float gh_factor=1.1,
                 pruning=None, preprocessing=None,
                 dump_gso_filename=None):

        if block_size <= 0:
            raise ValueError("block size must be > 0")
        if max_loops < 0:
            raise ValueError("maximum number of loops must be >= 0")
        if max_time < 0:
            raise ValueError("maximum time must be >= 0")
        if gh_factor <= 0:
            raise ValueError("GH factor must be <= 0")

        check_delta(delta)
        cdef BKZParam_c *o = new BKZParam_c(block_size, delta)

        cdef int linear_pruning_level = 0
        try:
            linear_pruning_level = int(pruning)
            if linear_pruning_level:
                o.enableLinearPruning(linear_pruning_level)
        except TypeError:
            if pruning:
                o.pruning.resize(block_size)
                for j in range(block_size):
                    o.pruning[j] = pruning[j]

        o.flags = flags

        if o.flags & BKZ_GH_BND:
            o.ghFactor = float(gh_factor)

        if auto_abort is True:
            o.flags |= BKZ_AUTO_ABORT

        if o.flags & BKZ_AUTO_ABORT:
            if auto_abort in (True, None):
                pass
            else:
                try:
                    a_scale, a_max = auto_abort
                    o.autoAbort_scale = a_scale
                    o.autoAbort_maxNoDec = a_max
                except TypeError:
                    del o
                    raise ValueError("Parameter auto_abort (%s) not understood."%auto_abort)

        if o.flags & BKZ_MAX_LOOPS:
            o.maxLoops = max_loops

        if o.flags & BKZ_MAX_TIME:
            o.maxTime = max_time

        if dump_gso_filename is not None:
            o.flags |= BKZ_DUMP_GSO

        if o.flags & BKZ_DUMP_GSO:
            o.dumpGSOFilename = dump_gso_filename

        if preprocessing:
            self.preprocessing = preprocessing
            o.preprocessing = self.preprocessing.o
        self.o = o

    # def __str__(self):

    #     cdef int block_size =
    #     cdef float delta =
    #     cdef int flags =
    #     cdef int max_loops =
    #     cdef int max_time =
    #     auto_abort =
    #     cdef float gh_factor =
    #     cdef pruning =
    #     cdef preprocessing =
    #     cdef dump_gso_filename =

    #     return "BKZParam()"

    def __dealloc__(self):
        del self.o

cdef class BKZAutoAbort:
    """
    """
    def __init__(self, MatGSO M, int num_rows, int start_row=0):
        """FIXME! briefly describe function

        :param MatGSO M:
        :param int num_rows:
        :param int start_row:
        :returns:
        :rtype:

        """
        if M._type == mpz_double:
            self._type = mpz_double
            self._core.mpz_double = new BKZAutoAbort_c[FP_NR[double]](M._core.mpz_double[0],
                                                                      num_rows,
                                                                      start_row)
        elif M._type == mpz_mpfr:
            self._type = mpz_mpfr
            self._core.mpz_mpfr = new BKZAutoAbort_c[FP_NR[mpfr_t]](M._core.mpz_mpfr[0],
                                                                  num_rows,
                                                                  start_row)
        else:
            raise RuntimeError("BKZAutoAbort object '%s' has no core."%self)

    def test_abort(self, scale=1.0, int max_no_dec=5):
        """FIXME! briefly describe function

        :param scale:
        :param int max_no_dec:
        :returns:
        :rtype:

        """
        if self._type == mpz_double:
            return self._core.mpz_double.testAbort(scale, max_no_dec)
        elif self._type == mpz_mpfr:
            return self._core.mpz_mpfr.testAbort(scale, max_no_dec)
        else:
            raise RuntimeError("BKZAutoAbort object '%s' has no core."%self)


# TODO: translate to a more fpLLL style interface

def bkz_reduction(IntegerMatrix A, int block_size, double delta=LLL_DEF_DELTA,
                  float_type=None, int precision=0, int max_loops=0, int max_time=0,
                  verbose=False, no_lll=False, bounded_lll=False, auto_abort=False, prune=False,
                  gh_bound=False, preprocessing=None, dump_gso_filename=None):
    r"""
    Run BKZ reduction.

    INPUT:

    :param block_size: an integer from 1 to ``nrows``
    :param delta: LLL parameter `0.25 < δ < 1.0`
    :param float_type: see list below for options

        - ``None``: for automatic choice

        - ``'double'``: double precision

        - ``'long double'``: long double precision

        - ``'dpe'``: double plus exponent

        - ``'mpfr'``: arbitrary precision, use ``precision`` to set desired bit length

    :param verbose: be verbose
    :param no_lll: disable LLL
    :param bounded_lll: bounded LLL
    :param precision: bit precision to use if ``fp`` is ``'mpfr'``
    :param max_loops: maximum number of full loops
    :param max_time: stop after time seconds (up to loop completion)
    :param auto_abort: heuristic, stop when the average slope of `\log(||b_i^*||)` does not decrease
        fast enough.  If a tuple is given it is parsed as ``(scale, max_iter)`` such that the
        algorithm will terminate if for ``max_iter`` loops the slope is not smaller than ``scale *
        old_slope`` where ``old_slope`` was the old minimum.  If ``True`` is given, this is
        equivalent to providing ``(1.0,5)`` which is fpLLL's default.
    :param prune: pruning parameter.
    :param gh_bound: heuristic, if ``True`` then the enumeration bound will be set to ``gh_bound``
        times the Gaussian Heuristic.  If ``True`` then gh_bound is set to 1.1, which is fpLLL's
        default.
    :param preprocessing:- if not ``None`` this is parameter is interpreted as a list of
        preprocessing options.  The following options are supported.

            - ``None``: LLL is run for pre-processing local blocks.

            - an integer: this is interpreted as the block size used for preprocessing local blocks
              before calling enumeration.  Any integer ≤ 2 disables BKZ preprocessing and runs LLL
              instead, any integer ≥ ``block_size`` raises an error.

            - an iterable of integers: this is interpreted as a list of pre-processing block sizes.
              For example, ``preprocessing=[20,5]`` would pre-process with BKZ-20 where blocks in
              turn are preprocessed with BKZ-5.

            - an iterable of tuples: this is interpreted as a list of pre-processing options where
              each entry ``(block_size, max_loops, max_time, auto_abort, prune)``.  It is
              permissable to not set all parameters.  For example, ``[(20,10)]`` is interpreted as
              ``[(20,0,0,True,0)]``.

    :param dump_gso_filename:- if this is not ``None`` then the logs of the norms of the
        Gram-Schmidt vectors are written to this file after each BKZ loop.
    """
    if block_size <= 0:
        raise ValueError("block size must be > 0")
    if max_loops < 0:
        raise ValueError("maximum number of loops must be >= 0")
    if max_time < 0:
        raise ValueError("maximum time must be >= 0")

    check_delta(delta)
    check_precision(precision)

    cdef FloatType floatType = check_float_type(float_type)

    cdef BKZParam_c o = BKZParam_c(block_size, delta)

    cdef int linear_pruning_level = 0
    try:
        linear_pruning_level = int(prune)
        if linear_pruning_level:
            o.enableLinearPruning(linear_pruning_level)
    except TypeError:
        if prune:
            o.pruning.resize(block_size)
            for j in range(block_size):
                o.pruning[j] = prune[j]

    o.preprocessing = NULL

    if verbose:
        o.flags |= BKZ_VERBOSE

    if no_lll:
        o.flags |= BKZ_NO_LLL

    if bounded_lll:
        o.flags |= BKZ_BOUNDED_LLL

    if gh_bound:
        o.flags |= BKZ_GH_BND
        if gh_bound is True:
            o.ghFactor = 1.1
        else:
            o.ghFactor = float(gh_bound)

    if auto_abort:
        o.flags |= BKZ_AUTO_ABORT
        try:
            a_s, a_l = auto_abort
            o.autoAbort_scale = a_s
            o.autoAbort_maxNoDec = a_l
        except TypeError:
            pass

    if max_loops:
        o.flags |= BKZ_MAX_LOOPS
        o.maxLoops = max_loops

    if max_time:
        o.flags |= BKZ_MAX_TIME
        o.maxTime = max_time

    if dump_gso_filename is not None:
        o.flags |= BKZ_DUMP_GSO
        o.dumpGSOFilename = dump_gso_filename

    cdef BKZParam_c *preproc = &o

    # preprocessing is None or False
    if not preprocessing:
        preprocessing = []

    # preprocessing is an integer
    try:
        _ = int(preprocessing)
        preprocessing = [preprocessing]
    except TypeError:
        pass

    # preprocessing is a list of integers
    try:
        _ = [int(step) for step in preprocessing]
        preprocessing = [(step,) for step in preprocessing]
    except TypeError:
        pass

    # preprocessing has shorter tuples in it
    tmp = []

    defaults = [None, 0, 0, True, 0]
    for step in preprocessing:
        if len(step) < 5:
            step = list(step) + defaults[len(step):]
            tmp.append(step)
    preprocessing = tmp

    cdef int flags
    if preprocessing:
        for i,step in enumerate(preprocessing):
            flags = BKZ_DEFAULT
            b, ml, mt, aa, prune = step

            if b <= 2:
                break
            if b > preproc.blockSize:
                raise ValueError("Preprocessing block size must be smaller than block size")

            a_s, a_l = 1.0,5
            try:
                a_s, a_l = aa
                flags = flags|BKZ_AUTO_ABORT
            except TypeError:
                if (aa is 0) or aa:
                    flags = flags|BKZ_AUTO_ABORT
            else:
                flags = flags|BKZ_AUTO_ABORT

            linearPruningLevel = 0
            prune = None
            try:
                linearPruningLevel = int(prune)
            except TypeError:
                pass

            preproc.preprocessing = new BKZParam_c(b, LLL_DEF_DELTA, flags, maxLoops=ml, maxTime=mt,
                                                   linearPruningLevel=linearPruningLevel,
                                                   autoAbort_scale=a_s, autoAbort_maxNoDec=a_l)
            preproc = preproc.preprocessing

            if prune:
                preproc.pruning.resize(b)
                for j in range(b):
                    preproc.pruning[j] = prune[j]

    sig_on()
    cdef int r = bkzReduction(A._core, NULL, o, floatType, precision)
    sig_off()

    recursively_free_bkz_param(o.preprocessing)

    if r:
        if r in (RED_BKZ_LOOPS_LIMIT, RED_BKZ_TIME_LIMIT):
            if verbose:
                print str(getRedStatusStr(r))
        else:
            raise ReductionError( str(getRedStatusStr(r)) )
