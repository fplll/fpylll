# -*- coding: utf-8 -*-
"""
Parameters for Block Korkine Zolotarev algorithm.

..  moduleauthor:: Martin R.  Albrecht <martinralbrecht+fpylll@googlemail.com>
"""

include "fpylll/config.pxi"

from cysignals.signals cimport sig_on, sig_off

from .fplll cimport BKZParam as BKZParam_c
from .fplll cimport BKZ_MAX_LOOPS, BKZ_MAX_TIME, BKZ_DUMP_GSO, BKZ_DEFAULT
from .fplll cimport BKZ_VERBOSE, BKZ_NO_LLL, BKZ_BOUNDED_LLL, BKZ_GH_BND, BKZ_AUTO_ABORT
from .fplll cimport BKZ_DEF_AUTO_ABORT_SCALE, BKZ_DEF_AUTO_ABORT_MAX_NO_DEC
from .fplll cimport BKZ_DEF_GH_FACTOR, BKZ_DEF_MIN_SUCCESS_PROBABILITY
from .fplll cimport BKZ_DEF_RERANDOMIZATION_DENSITY
from .fplll cimport PRUNER_METRIC_PROBABILITY_OF_SHORTEST
from .fplll cimport PRUNER_METRIC_EXPECTED_SOLUTIONS
from .fplll cimport LLL_DEF_DELTA
from .fplll cimport PruningParams as PruningParams_c
from .fplll cimport Strategy as Strategy_c
from .fplll cimport load_strategies_json as load_strategies_json_c
from .fplll cimport strategy_full_path

from fpylll.util cimport check_delta, check_pruner_metric
from cython.operator cimport dereference as deref, preincrement as inc

from fpylll.config import default_strategy_path, default_strategy

from .pruner cimport PruningParams

from collections import OrderedDict
from os import path, environ

import json

cdef class Strategy:
    """
    A strategy is a collection of pruning coefficients for a variety
    of radii and preprocessing block sizes.
    """
    def __init__(self, block_size, preprocessing_block_sizes=tuple(), pruning_parameters=tuple(), **kwds):
        """
        :param block_size: block size of this strategy
        :param preprocessing_block_sizes: preprocessing block sizes
        :param pruning_parameters: a list of pruning parameters

        All other keyword arguments are stored as auxiliary parameters in the ``aux`` attribute.

        """

        if block_size < 0:
            raise ValueError("Block size must be ≥ 0")
        self._core.block_size = block_size

        for p in pruning_parameters:
            if not isinstance(p, PruningParams):
                p = PruningParams(*p)
            self._core.pruning_parameters.push_back((<PruningParams>p)._core)

        if len(pruning_parameters) == 0:
            p = PruningParams(1.0, [1.0 for _ in range(self.block_size)], 1.0)
            self._core.pruning_parameters.push_back((<PruningParams>p)._core)

        for p in preprocessing_block_sizes:
            if p<=2:
                raise ValueError("Preprocessing block_size must be > 2, got %s", p)
            if p >= block_size:
                raise ValueError("Preprocessing block_size must be < block size, got %s", p)
            self._core.preprocessing_block_sizes.push_back(p)

        self.aux = {}
        for k,v in kwds.items():
            self.aux[k] = v

    def get_pruning(self, radius, gh):
        """

        :param radius: target radius
        :param gh:     gaussian heuristic radius

        """
        gh_factor = radius/gh
        closest_dist = 2**80
        best = None
        for pruning in self.pruning_parameters:
            if abs(pruning.gh_factor - gh_factor) < closest_dist:
                best = pruning
                closest_dist = abs(pruning.gh_factor - gh_factor)
        assert(best is not None)
        return best

    def dict(self):
        """

            >>> from fpylll import load_strategies_json, BKZ
            >>> print(load_strategies_json(BKZ.DEFAULT_STRATEGY)[50].dict()) # doctest: +ELLIPSIS
            OrderedDict...
            >>> print(load_strategies_json(BKZ.DEFAULT_STRATEGY)[50])
            Strategy< 50, (26), 0.52-0.55, {}>

        """
        d = OrderedDict()
        d["block_size"] = self.block_size
        d["preprocessing_block_sizes"] = self.preprocessing_block_sizes
        d["pruning_parameters"] = tuple([(p.gh_factor, p.coefficients, p.expectation, p.metric, p.detailed_cost)
                                         for p in self.pruning_parameters])
        for k,v in self.aux.items():
            d[k] = v
        return d

    def __getitem__(self, what):
        """

            >>> from fpylll.fplll.bkz_param import Strategy
            >>> s = Strategy(20, [10], [], foo=True)
            >>> s["foo"]
            True

        """
        return self.aux[what]

    def __str__(self):
        preproc = ",".join([str(p) for p in self.preprocessing_block_sizes])
        pruning = [p.expectation for p in self.pruning_parameters]
        if pruning:
            pruning = min(pruning), max(pruning)
        else:
            pruning = 1.0, 1.0
        return "Strategy<%3d, (%s), %4.2f-%4.2f, %s>"%(self.block_size, preproc, pruning[0], pruning[1], self.aux)

    def __reduce__(self):
        """
            >>> from fpylll import Pruning
            >>> from fpylll.fplll.bkz_param import Strategy
            >>> import pickle
            >>> p = Pruning.PruningParams(1.0, [1.0, 0.75, 0.5, 0.25], 0.5)
            >>> print(pickle.loads(pickle.dumps(Strategy(20, [10], [p]))))
            Strategy< 20, (10), 0.50-0.50, {}>

        """
        return unpickle_Strategy, (self.__class__, tuple(self.dict().items()))

    @staticmethod
    cdef Strategy from_cxx(Strategy_c& s):
        cdef Strategy self = Strategy(0)
        self._core = s
        return self

    @staticmethod
    cdef to_cxx(Strategy_c& self, Strategy s):

        for p in s.pruning_parameters:
            self.pruning_parameters.push_back((<PruningParams>p)._core)

        for p in s.preprocessing_block_sizes:
            self.preprocessing_block_sizes.push_back(p)

        self.block_size = s.block_size

    @property
    def block_size(self):
        """
            >>> from fpylll import Pruning
            >>> from fpylll.fplll.bkz_param import Strategy
            >>> s = Strategy(20, [10], [Pruning.PruningParams(1.0, [1.0, 0.75, 0.5, 0.25], 0.5)])
            >>> s.block_size
            20

        """
        return self._core.block_size

    @property
    def preprocessing_block_sizes(self):
        """
            >>> from fpylll import Pruning
            >>> from fpylll.fplll.bkz_param import Strategy
            >>> s = Strategy(20, [10], [Pruning.PruningParams(1.0, [1.0, 0.75, 0.5, 0.25], 0.5)])
            >>> s.preprocessing_block_sizes
            (10,)

        """
        cdef list preprocessing_block_sizes = []
        cdef vector[size_t].iterator it = self._core.preprocessing_block_sizes.begin()
        while it != self._core.preprocessing_block_sizes.end():
            preprocessing_block_sizes.append(deref(it))
            inc(it)
        return tuple(preprocessing_block_sizes)

    @property
    def pruning_parameters(self):
        """
            >>> from fpylll import Pruning
            >>> from fpylll.fplll.bkz_param import Strategy
            >>> s = Strategy(20, [10], [Pruning.PruningParams(1.0, [1.0, 0.75, 0.5, 0.25], 0.5)])
            >>> print(s.pruning_parameters[0])
            PruningParams<1.000000, (1.00,...,0.25), 0.5000>

        """
        cdef list pruning_parameters = []
        cdef vector[PruningParams_c].iterator it = self._core.pruning_parameters.begin()
        while it != self._core.pruning_parameters.end():
            pruning_parameters.append(PruningParams.from_cxx(deref(it)))
            inc(it)
        return tuple(pruning_parameters)


cdef strategies_c_to_strategies(vector[Strategy_c]& strategies):
    """
    Convert C++ strategy vector to Python strategy list
    """
    cdef vector[Strategy_c].iterator it = strategies.begin()
    ret = []
    while it != strategies.end():
        ret.append(Strategy.from_cxx(deref(it)))
        inc(it)
    return tuple(ret)

cdef _load_strategies_aux_json(strategies, filename):
    """
    Load auxiliary data into strategies.

    We call ``load_strategies_json_c`` ignores all attributes not known at the C++ level.  Thus, we
    read the file twice to load those auxiliary attributes.  It's a bit hamfisted but it works and
    allows us to avoid reimplementing ``load_strategies_json_c`` here.

    ::

        >>> import tempfile
        >>> from fpylll.fplll.bkz_param import Strategy, load_strategies_json, dump_strategies_json
        >>> fh, fn = tempfile.mkstemp(suffix=".json")
        >>> s = [Strategy(0, [], [], foo=True)]
        >>> dump_strategies_json(fn, s)
        >>> s = load_strategies_json(fn)
        >>> s[0]["foo"]
        True

    """
    with open(filename, "r") as fh:
        json_ = json.load(fh)
        for i in range(len(strategies)):
            if not isinstance(strategies[i], Strategy):
                raise TypeError("{i}-th input {strategy} is not a Strategy".format(i=i, strategy=strategies[i]))
            for k,v in json_[i].items():
                if not hasattr(strategies[i], k):
                    (<Strategy>strategies[i]).aux[k] = v
    return strategies

def load_strategies_json(filename):
    """
    Load strategies from ``filename``.

    ::

        >>> from fpylll import load_strategies_json, BKZ
        >>> strategies = load_strategies_json(BKZ.DEFAULT_STRATEGY)
        >>> strategies[80].preprocessing_block_sizes
        (56,)

        >>> strategies[80].pruning_parameters[0].expectation
        0.14362574329237188

    """

    #  It turns out that on some platforms (e.g pre-compiled Sagemath) the BKZ.DEFAULT_STRATEGY
    #  path gets a bit broken. If this happens there's a C++ exception thrown from fplll, which we
    #  catch. If it just so happens that we fail, we read the Sagemath environment variables and try
    #  to navigate to the default file relative to that directory.

    if not path.exists(filename) and filename == "default.json":
        if 'SAGE_LOCAL' in environ:
            # NOTE: some people are quite angry about this: https://github.com/fplll/fpylll/issues/221
            filename = path.join(environ['SAGE_LOCAL'], 'fplll', 'strategies', 'default.json')
        else:
            filename = path.join(default_strategy_path, filename)

    if not path.exists(filename):
        raise FileNotFoundError("File '%s' not found."%filename)

    if isinstance(filename, bytes):
        filename = filename.decode("UTF-8")
    if isinstance(filename, (str, unicode)):
        filename = filename.encode('UTF-8')

    cdef vector[Strategy_c] strategies

    sig_on()
    strategies = load_strategies_json_c(filename)
    sig_off()

    strategies_ = strategies_c_to_strategies(strategies)
    strategies_ = _load_strategies_aux_json(strategies_, filename)
    return strategies_


def dump_strategies_json(filename, strategies):
    with open(filename, "w") as fh:
        json.dump([strategy.dict() for strategy in strategies], fh, indent=4, sort_keys=True)

cdef load_strategies_python(vector[Strategy_c]& out, inp):
    for strategy in inp:
        if isinstance(strategy, OrderedDict):
            strategy = Strategy(**strategy)
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
                 auto_abort=None,
                 gh_factor=None,
                 float min_success_probability=BKZ_DEF_MIN_SUCCESS_PROBABILITY,
                 int rerandomization_density=BKZ_DEF_RERANDOMIZATION_DENSITY,
                 dump_gso_filename=None, **kwds):
        """
        Create BKZ parameters object.

        :param block_size: an integer from 1 to ``nrows``
        :param strategies: a filename or a list of Strategies
        :param delta: LLL parameter `0.25 < δ < 1.0`
        :param flags: flags
        :param max_loops: maximum number of full loops
        :param max_time: stop after time seconds (up to loop completion)
        :param auto_abort: heuristic, stop when the average slope of `\log(||b_i^*||)` does not
            decrease fast enough.  If a tuple is given it is parsed as ``(scale, max_iter)`` such
            that the algorithm will terminate if for ``max_iter`` loops the slope is not smaller
            than ``scale * old_slope`` where ``old_slope`` was the old minimum.  If ``True`` is
            given, this is equivalent to providing ``(1.0,5)`` which is fpLLL's default.
        :param gh_factor: heuristic, if set then the enumeration bound will be set to ``gh_factor``
            times the Gaussian Heuristic.  If ``True`` then ``gh_factor`` is set to 1.1, which is
            fpLLL's default.
        :param min_success_probability: minimum success probability in an SVP reduction (when using
            pruning)
        :param rerandomization_density: density of rerandomization operation when using extreme
            pruning
        :param dump_gso_filename: if this is not ``None`` then the logs of the norms of the
            Gram-Schmidt vectors are written to this file after each BKZ loop.

        All other keyword arguments are stored as auxiliary parameters in
        the ``aux`` attribute.
        """

        # if the user sets these, they want the appropriate flags to be set
        if max_loops > 0:
            flags |= BKZ_MAX_LOOPS
        if max_time > 0:
            flags |= BKZ_MAX_TIME
        if gh_factor is not None:
            flags |= BKZ_GH_BND
        if gh_factor in (True, False, None):
            gh_factor = BKZ_DEF_GH_FACTOR

        if block_size <= 0:
            raise ValueError("block size must be > 0")
        if max_loops < 0:
            raise ValueError("maximum number of loops must be >= 0")
        if max_time < 0:
            raise ValueError("maximum time must be >= 0")
        if gh_factor <= 0:
            raise ValueError("GH factor must be > 0")

        check_delta(delta)
        if strategies:
            if isinstance(strategies, bytes):
                strategies = strategies.decode("UTF-8")
            if isinstance(strategies, (str, unicode)):
                strategies = strategies.encode('UTF-8')
                sig_on()
                self.strategies_c = load_strategies_json_c(strategy_full_path(strategies))
                self.strategies = strategies_c_to_strategies(self.strategies_c)
                sig_off()
            else:
                load_strategies_python(self.strategies_c, strategies)
                if all(isinstance(x, Strategy) for x in strategies):
                    self.strategies = tuple(strategies)
                elif all(isinstance(x, OrderedDict) for x in strategies):
                    self.strategies = tuple([Strategy(**strategy) for strategy in strategies])
                else:
                    raise TypeError("Entry type of strategies must be Strategy or OrderedDict")

        cdef BKZParam_c *o = new BKZParam_c(block_size, self.strategies_c, delta)

        if not strategies:
            self.strategies = strategies_c_to_strategies(o.strategies)

        o.flags = flags
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
                    raise ValueError("Parameter auto_abort (%s) not understood."%auto_abort)

        if o.flags & BKZ_MAX_LOOPS:
            o.max_loops = max_loops

        if o.flags & BKZ_MAX_TIME:
            o.max_time = max_time

        if dump_gso_filename is not None:
            o.flags |= BKZ_DUMP_GSO

        if o.flags & BKZ_DUMP_GSO and dump_gso_filename is not None:
            o.dump_gso_filename = dump_gso_filename

        o.min_success_probability = min_success_probability
        o.rerandomization_density = rerandomization_density

        self.o = o
        self.aux = {}
        for k,v in kwds.iteritems():
            self.aux[k] = v

    def __dealloc__(self):
        del self.o

    def __repr__(self):
        return "<BKZParam(%d, flags=0x%04x) at %s>" % (
            self.o.block_size,
            self.o.flags,
            hex(id(self)))

    def __reduce__(self):
        return unpickle_BKZParam, tuple(self.dict().items())

    def __str__(self):
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
    def dump_gso_filename(self):
        return self.o.dump_gso_filename

    @property
    def min_success_probability(self):
        return self.o.min_success_probability

    @property
    def rerandomization_density(self):
        return self.o.rerandomization_density

    def __getitem__(self, what):
        """

            >>> from fpylll import BKZ
            >>> p = BKZ.Param(40, max_loops=4, foo=True)
            >>> p["foo"]
            True

        """
        return self.aux[what]

    def __setitem__(self, what, value):
        """

            >>> from fpylll import BKZ
            >>> p = BKZ.Param(40, max_loops=4, foo=True)
            >>> p["foo"] = False
            >>> p["foo"]
            False

        """
        if not isinstance(what, str):
            raise TypeError("Only strings are supported as auxiliary keys but got %s"%type(what))
        if hasattr(self, what):
            raise ValueError("Provided key '%s' would shadow class attribute"%what)
        self.aux[what] = value

    def dict(self, all=True):
        """

            >>> from fpylll import BKZ
            >>> d = BKZ.Param(40, max_loops=4, flags=BKZ.MAX_LOOPS).dict(False)
            >>> d["block_size"]
            40
            >>> d["max_loops"]
            4
            >>> d.get("delta", False)
            False

        """
        d = {}
        d["block_size"] = self.block_size
        if all or abs(self.delta - LLL_DEF_DELTA) > 0.001:
            d["delta"] = self.delta
        d["flags"] = self.flags
        if all or self.max_loops != 0:
            d["max_loops"] = self.max_loops
        if all or self.max_time != 0:
            d["max_time"] = self.max_time
        if self.o.flags & BKZ_AUTO_ABORT:
            d["auto_abort"] = self.auto_abort
        if self.o.flags & BKZ_GH_BND:
            d["gh_factor"] = self.gh_factor
        if self.o.flags & BKZ_DUMP_GSO:
            d["dump_gso_filename"] =  self.dump_gso_filename
        if all or self.min_success_probability != BKZ_DEF_MIN_SUCCESS_PROBABILITY:
            d["min_success_probability"] = self.min_success_probability
        if all or self.rerandomization_density != BKZ_DEF_RERANDOMIZATION_DENSITY:
            d["rerandomization_density"]  = self.rerandomization_density
        if all:
            d["strategies"] = [strategy.dict() for strategy in self.strategies[:self.block_size+1]]
        if all:
            for k,v in self.aux.iteritems():
                d[k] = v

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
    >>> pickle.loads(pickle.dumps(BKZ.Param(10, flags=BKZ.VERBOSE))) # doctest: +ELLIPSIS
    <BKZParam(10, flags=0x0001) at 0x...>

    """
    kwds = dict(args)
    return BKZParam(**kwds)

def unpickle_Strategy(*args):
    """
    """
    cls, args = args
    kwds = dict(args)
    return cls(**kwds)


def BKZEasyParam(block_size, **kwds):
    """
    Set sane defaults for most use-cases:

        1. set default strategies

        2. switch on auto abort

    Additional parameters can be passed.  When flags are passed, they are XORed with the default
    flags.

    :param block_size: BKZ block size

    """
    easy_kwds = {}
    easy_kwds["block_size"] = block_size
    easy_kwds["strategies"] = default_strategy
    easy_kwds["flags"] = BKZ_DEFAULT|BKZ_AUTO_ABORT

    for k,v in kwds.iteritems():
        if k == "flags":
            easy_kwds[k] ^= v
        else:
            easy_kwds[k] = v
    return BKZParam(**easy_kwds)
