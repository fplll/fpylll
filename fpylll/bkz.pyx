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

cdef class BKZParam:

    def __init__(self, int block_size, float delta=LLL_DEF_DELTA, int flags=BKZ_DEFAULT,
                 int max_loops=0, int max_time=0,
                 auto_abort=None, float gh_factor=1.1,
                 pruning=None, BKZParam preprocessing=None,
                 dump_gso_filename=None):
        """
        Create BKZ parameters object.

        :param block_size: an integer from 1 to ``nrows``
        :param delta: LLL parameter `0.25 < δ < 1.0`
        :param max_loops: maximum number of full loops
        :param max_time: stop after time seconds (up to loop completion)
        :param auto_abort: heuristic, stop when the average slope of `\log(||b_i^*||)` does not decrease
            fast enough.  If a tuple is given it is parsed as ``(scale, max_iter)`` such that the
            algorithm will terminate if for ``max_iter`` loops the slope is not smaller than ``scale *
            old_slope`` where ``old_slope`` was the old minimum.  If ``True`` is given, this is
            equivalent to providing ``(1.0,5)`` which is fpLLL's default.
        :param pruning: pruning parameter.
        :param gh_factor: heuristic, if ``True`` then the enumeration bound will be set to ``gh_bound``
            times the Gaussian Heuristic.  If ``True`` then gh_bound is set to 1.1, which is fpLLL's
            default.
        :param preprocessing:- if not ``None`` this is parameter is interpreted as a list of
            preprocessing options.  The following options are supported.

            - ``None``: LLL is run for pre-processing local blocks.

            - an BKZParam object

        :param dump_gso_filename:- if this is not ``None`` then the logs of the norms of the
            Gram-Schmidt vectors are written to this file after each BKZ loop.
        """

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
            if linear_pruning_level > block_size:
                raise ValueError("Linear pruning level (%d) bigger than block size (%d)"%(linear_pruning_level,
                                                                                          block_size))
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
            self._preprocessing = preprocessing
            o.preprocessing = self._preprocessing.o

        self.o = o

    def __dealloc__(self):
        del self.o

    def __repr__(self):
        return "<BKZParam(%d, flags=0x%04x) at %s>" % (
            self.o.blockSize,
            self.o.flags,
            hex(id(self)))

    def __str__(self):
        """FIXME! briefly describe function

        :returns:
        :rtype:

        """
        cdef BKZParam param = self
        i = 0
        r = []
        while param:
            if i > 0:
                prep = " "
            else:
                prep = ""
            r.append(prep + str(param.dict(all=False)))
            param = param.preprocessing
            i += 1
        r = "(" + ",\n".join(r) + ")"
        return r

    @property
    def block_size(self):
        return self.o.blockSize

    @property
    def delta(self):
        return self.o.delta

    @property
    def flags(self):
        return self.o.flags

    @property
    def max_loops(self):
        return self.o.maxLoops

    @property
    def max_time(self):
        return self.o.maxTime

    @property
    def auto_abort(self):
        a_scale = self.o.autoAbort_scale
        a_max = self.o.autoAbort_maxNoDec
        return (a_scale, a_max)

    @property
    def gh_factor(self):
        return self.o.ghFactor

    @property
    def pruning(self):
        p = [self.o.pruning[j] for j in range(len(self.o.pruning))]
        if p:
            return tuple(p)
        else:
            return None

    @property
    def preprocessing(self):
        return self._preprocessing

    @property
    def dump_gso_filename(self):
        return self.o.dumpGSOFilename

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise ValueError("Key '%s' not found."%key)


    def dict(self, all=True):
        """
        """
        d = {}
        d["block_size"] = self.block_size
        if all or self.delta == LLL_DEF_DELTA:
            d["delta"] = self.delta
        d["flags"] = self.flags
        if all or self.max_loops != 0:
            d["max_loops"] = self.max_loops
        if all or self.max_time != 0:
            d["max_time"] = self.max_time
        if all or self.auto_abort != (1.0, 5):
            d["auto_abort"] = self.auto_abort
        if all or self.gh_factor != 1.1:
            d["gh_factor"] = self.gh_factor
        if all or self.pruning:
            d["pruning"] = self.pruning
        if all or self.preprocessing:
            d["preprocessing"] =  self.preprocessing
        if all or self.dump_gso_filename != "gso.log":
            d["dump_gso_filename"] =  self.dump_gso_filename
        return d

    def new(self, **kwds):
        d = self.dict()
        d.update(kwds)
        return BKZParam(**d)

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


def bkz_reduction(IntegerMatrix A, BKZParam o, float_type=None, int precision=0):
    """
    Run BKZ reduction.

    :param o: BKZ parameters
    :param float_type: see list below for options

        - ``None``: for automatic choice

        - ``'double'``: double precision

        - ``'long double'``: long double precision

        - ``'dpe'``: double plus exponent

        - ``'mpfr'``: arbitrary precision, use ``precision`` to set desired bit length

    :param precision: bit precision to use if ``fp`` is ``'mpfr'``
    """
    check_precision(precision)

    cdef FloatType floatType = check_float_type(float_type)

    sig_on()
    cdef int r = bkzReduction(A._core, NULL, o.o[0], floatType, precision)
    sig_off()

    if r:
        if r in (RED_BKZ_LOOPS_LIMIT, RED_BKZ_TIME_LIMIT):
            pass
        else:
            raise ReductionError( str(getRedStatusStr(r)) )


class BKZ:
    DEFAULT = BKZ_DEFAULT
    VERBOSE = BKZ_VERBOSE
    NO_LLL = BKZ_NO_LLL
    BOUNDED_LLL = BKZ_BOUNDED_LLL,
    GH_BND = BKZ_GH_BND
    AUTO_ABORT = BKZ_AUTO_ABORT
    MAX_LOOPS = BKZ_MAX_LOOPS
    MAX_TIME = BKZ_MAX_TIME
    DUMP_GSO = BKZ_DUMP_GSO

    Param = BKZParam
    AutoAbort = BKZAutoAbort
    reduction = bkz_reduction
