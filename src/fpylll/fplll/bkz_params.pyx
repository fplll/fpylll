# -*- coding: utf-8 -*-
include "fpylll/config.pxi"
include "cysignals/signals.pxi"

"""
Parameters for Block Korkine Zolotarev algorithm.

..  moduleauthor:: Martin R.  Albrecht <martinralbrecht+fpylll@googlemail.com>
"""
from fplll cimport BKZParam as BKZParam_c
from fplll cimport BKZ_MAX_LOOPS, BKZ_MAX_TIME, BKZ_DUMP_GSO, BKZ_DEFAULT
from fplll cimport BKZ_VERBOSE, BKZ_NO_LLL, BKZ_BOUNDED_LLL, BKZ_GH_BND, BKZ_AUTO_ABORT
from fplll cimport LLL_DEF_DELTA
from fplll cimport Pruning as Pruning_c
from fplll cimport Strategy as Strategy_c
from fplll cimport load_strategies_json as load_strategies_json_c

from fpylll.util cimport check_delta
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref, preincrement as inc

cdef class Pruning:
    """
    Pruning parameters.
    """
    def __init__(self, radius_factor, coefficients, probability=1.0):
        """Create new pruning parameters object.

        :param radius_factor: ratio of radius to Gaussian heuristic
        :param coefficients:  a list of pruning coefficients
        :param probability:   success probability

        """
        if radius_factor <= 0:
            raise ValueError("Radius factor must be > 0")
        self.radius_factor = radius_factor
        self.coefficients = tuple(coefficients)
        if probability <= 0 or probability > 1:
            raise ValueError("Probability must be between 0 and 1")
        self.probability = probability

        Pruning.to_cxx(self._core, self)

    @staticmethod
    cdef Pruning from_cxx(Pruning_c& p):
        """
        Load Pruning object from C++ Pruning object.
        """
        self = Pruning()
        coefficients = []
        self.radius_factor = p.radius_factor
        self.probability = p.probability

        cdef vector[double].iterator it = p.coefficients.begin()
        while it != p.coefficients.end():
            coefficients.append(deref(it))
            inc(it)
        self.coefficients = tuple(coefficients)
        self._core = p
        return self

    @staticmethod
    cdef to_cxx(Pruning_c& self, Pruning p):
        """
        Store pruning object in C++ pruning object.
        """
        self.radius_factor = p.radius_factor
        self.probability = p.probability
        for c in p.coefficients:
            self.coefficients.push_back(c)

    @staticmethod
    def LinearPruning(block_size, level):
        """
        Set all pruning coefficients to 1, except the last <level>
        coefficients, these will be linearly with slope `-1 / block_size`.

        :param block_size: block size
        :param level: level
        """
        sig_on()
        cdef Pruning_c p = Pruning_c.LinearPruning(block_size, level)
        sig_off()
        return Pruning.from_cxx(p)

cdef class Strategy:
    """
    A strategy is a collection of pruning coefficients for a variety
    of radii and preprocessing block sizes.
    """
    def __init__(self, pruning_parameters, preprocessing_blocksizes):
        """

        :param pruning_parameters: a list of pruning parameters
        :param preprocessing_blocksizes: preprocessing block sizes

        """

        for p in pruning_parameters:
            if not isinstance(p, Pruning):
                p = Pruning(p)
            self.pruning_parameters.append(p)
        for p in preprocessing_blocksizes:
            if p<=2:
                raise ValueError("Preprocessing blocksize must be > 2, got %s", p)
            self.preprocessing_blocksizes.append(int(p))
        Strategy.to_cxx(self._core, self)


    @staticmethod
    cdef Strategy from_cxx(Strategy_c& s):
        self = Strategy()
        pruning_parameters = []

        cdef vector[Pruning_c].iterator pit = s.pruning_parameters.begin()
        while pit != s.pruning_parameters.end():
            pruning_parameters.append(Pruning.from_cxx(deref(pit)))
            inc(pit)
        self.pruning_parameters = tuple(pruning_parameters)

        preprocessing_blocksizes = []
        cdef vector[int].iterator bit = s.preprocessing_blocksizes.begin()
        while bit != s.preprocessing_blocksizes.end():
            preprocessing_blocksizes.append(deref(bit))
            inc(bit)
        self.preprocessing_blocksizes = tuple(preprocessing_blocksizes)
        return self

    @staticmethod
    cdef to_cxx(Strategy_c& self, Strategy s):

        for p in s.pruning_parameters:
            self.pruning_parameters.push_back((<Pruning>p)._core)

        for p in s.preprocessing_blocksizes:
            self.preprocessing_blocksizes.push_back(p)


cdef strategies_c_to_strategies(vector[Strategy_c]& strategies):
    cdef vector[Strategy_c].iterator it = strategies.begin()
    ret = []
    while it != strategies.end():
        ret.append(Strategy.from_cxx(deref(it)))
        inc(it)
    return tuple(ret)


def load_strategies_json(filename):
    cdef vector[Strategy_c] strategies
    sig_on()
    strategies = load_strategies_json_c(filename)
    sig_off()
    return strategies_c_to_strategies(strategies)


cdef load_strategies_python(vector[Strategy_c]& out, inp):
    for strategy in inp:
        if not isinstance(strategy, Strategy):
            raise TypeError("Type '%s' of '%s' not supported."%(type(strategy), strategy))
        out.push_back((<Strategy>strategy)._core)


cdef class BKZParam:
    """
    Parameters for the BKZ algorithm.
    """
    def __init__(self, int block_size, strategies=None,
                 float delta=LLL_DEF_DELTA, int flags=BKZ_DEFAULT,
                 int max_loops=0, int max_time=0,
                 auto_abort=None, float gh_factor=1.1,
                 dump_gso_filename=None):
        """
        Create BKZ parameters object.

        :param block_size: an integer from 1 to ``nrows``
        :param strategies: a filename or a list of Strategies
        :param delta: LLL parameter `0.25 < δ < 1.0`
        :param max_loops: maximum number of full loops
        :param max_time: stop after time seconds (up to loop completion)
        :param auto_abort: heuristic, stop when the average slope of `\log(||b_i^*||)` does not
            decrease fast enough.  If a tuple is given it is parsed as ``(scale, max_iter)`` such
            that the algorithm will terminate if for ``max_iter`` loops the slope is not smaller
            than ``scale * old_slope`` where ``old_slope`` was the old minimum.  If ``True`` is
            given, this is equivalent to providing ``(1.0,5)`` which is fpLLL's default.
        :param gh_factor: heuristic, if ``True`` then the enumeration bound will be set to
            ``gh_factor`` times the Gaussian Heuristic.  If ``True`` then ``gh_factor`` is set to 1.1,
            which is fpLLL's default.
        :param dump_gso_filename: if this is not ``None`` then the logs of the norms of the
            Gram-Schmidt vectors are written to this file after each BKZ loop.

        """

        if block_size <= 0:
            raise ValueError("block size must be > 0")
        if max_loops < 0:
            raise ValueError("maximum number of loops must be >= 0")
        if max_time < 0:
            raise ValueError("maximum time must be >= 0")
        if gh_factor <= 0:
            raise ValueError("GH factor must be > 0")

        check_delta(delta)
        cdef vector[Strategy_c] *strategies_c = new vector[Strategy_c]()

        if strategies:
            if isinstance(strategies, str):
                sig_on()
                strategies_c[0] = load_strategies_json_c(strategies)
                sig_off()
            else:
                load_strategies_python(strategies_c[0], strategies)

        cdef BKZParam_c *o = new BKZParam_c(block_size, strategies_c[0], delta)

        o.flags = flags

        if o.flags & BKZ_GH_BND:
            o.gh_factor = float(gh_factor)

        if auto_abort is True:
            o.flags |= BKZ_AUTO_ABORT

        if o.flags & BKZ_AUTO_ABORT:
            if auto_abort in (True, None):
                pass
            else:
                try:
                    a_scale, a_max = auto_abort
                    o.auto_abort_scale = a_scale
                    o.auto_abort_max_no_dec = a_max
                except TypeError:
                    del o
                    del strategies_c
                    raise ValueError("Parameter auto_abort (%s) not understood."%auto_abort)

        if o.flags & BKZ_MAX_LOOPS:
            o.max_loops = max_loops

        if o.flags & BKZ_MAX_TIME:
            o.max_time = max_time

        if dump_gso_filename is not None:
            o.flags |= BKZ_DUMP_GSO

        if o.flags & BKZ_DUMP_GSO:
            o.dump_gso_filename = dump_gso_filename

        self.o = o

    def __dealloc__(self):
        del self.o

    def __repr__(self):
        return "<BKZParam(%d, flags=0x%04x) at %s>" % (
            self.o.block_size,
            self.o.flags,
            hex(id(self)))

    def __reduce__(self):
        """
        Make sure attempts at pickling raise an error until proper pickling is implemented.
        """
        return unpickle_BKZParam, tuple(self.dict().items())

    def __str__(self):
        """FIXME! briefly describe function

        :returns:
        :rtype:

        """
        cdef BKZParam param = self
        r = str(param.dict(all=False))
        return r

    @property
    def block_size(self):
        return self.o.block_size

    @property
    def delta(self):
        return self.o.delta

    @property
    def flags(self):
        return self.o.flags

    @property
    def max_loops(self):
        return self.o.max_loops

    @property
    def max_time(self):
        return self.o.max_time

    @property
    def auto_abort(self):
        a_scale = self.o.auto_abort_scale
        a_max = self.o.auto_abort_max_no_dec
        return (a_scale, a_max)

    @property
    def gh_factor(self):
        return self.o.gh_factor

    @property
    def preprocessing(self):
        return self._preprocessing

    @property
    def dump_gso_filename(self):
        return self.o.dump_gso_filename

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
        if all or self.dump_gso_filename != "gso.log":
            d["dump_gso_filename"] =  self.dump_gso_filename
        return d

    def new(self, **kwds):
        d = self.dict()
        d.update(kwds)
        return BKZParam(**d)


def unpickle_BKZParam(*args):
    """
    Deserialize this set of BKZ parameters.

    >>> from fpylll import BKZ
    >>> import pickle
    >>> pickle.loads(pickle.dumps(BKZ.Param(10, ))) # doctest: +ELLIPSIS
    <BKZParam(10, flags=0x0040) at 0x...>

    """
    kwds = dict(args)
    return BKZParam(**kwds)
