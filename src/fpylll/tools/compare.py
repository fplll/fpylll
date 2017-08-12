# -*- coding: utf-8 -*-
"""
Compare the performance of BKZ variants.

..  moduleauthor:: Martin R.  Albrecht <martinralbrecht+fpylll@googlemail.com>

..  note :: This module does not work with standard BKZ classes.  Instead, it expects that the BKZ
classes it consumes accept a block size as ``params`` in the ``tour`` member function.  This way,
the construction of param objects can be rolled into the class description. See example classes below.
"""


# Imports

from __future__ import absolute_import
from collections import OrderedDict
from fpylll import IntegerMatrix, BKZ
from fpylll import set_random_seed
from fpylll.tools.bkz_stats import BKZTreeTracer, dummy_tracer, pretty_dict
from fpylll.tools.quality import basis_quality
from fpylll.util import ReductionError

from multiprocessing import Pool
import logging
import copy
import time

import fpylll.algorithms.bkz
import fpylll.algorithms.bkz2
import fpylll.algorithms.bkz2_otf
import fpylll.algorithms.bkz2_otf_subsol


# Utility Functions

def bkz_call(BKZ, A, block_size, tours, progressive_step_size=None):
    """Call ``BKZ`` on ``A`` with ``block_size`` for the given number of ``tours``.

    If ``return_queue`` is not ``None`` then the trace and the provided ``tag`` are put on the
    queue.  Otherwise, they are returned.

    :param BKZ:
    :param A:
    :param block_size:
    :param tours:
    :param progressive_step_size:

    .. note :: This function essentially reimplements ``BKZ.__call__`` but supports the progressive strategy.

    """
    bkz = BKZ(copy.copy(A))
    tracer = BKZTreeTracer(bkz, start_clocks=True)

    # this essentially initialises the GSO object, LLL was already run by the constructor, so this
    # is quick.
    with tracer.context("lll"):
        bkz.lll_obj()

    if progressive_step_size is None:
        block_sizes = (block_size,)
    elif int(progressive_step_size) > 0:
        block_sizes = range(2, block_size+1, progressive_step_size)
        if block_sizes[-1] != block_size:
            block_sizes.append(block_size)
    else:
        raise ValueError("Progressive step size of %s not understood."%progressive_step_size)

    for block_size in block_sizes:
        for i in range(tours):
            with tracer.context("tour", (block_size, i)):
                bkz.tour(block_size, tracer=tracer)
    tracer.exit()
    trace = tracer.trace

    quality = basis_quality(bkz.M)
    for k, v in quality.items():
        trace.data[k] = v

    return trace


class CompareBKZ:
    def __init__(self, classes, matrixf, dimensions, block_sizes, progressive_step_size, log_filename=None):
        """
        :param classes: a list of BKZ classes to test.  See caveat above.
        :param matrixf: A function to create matrices for a given dimension and block size
        :param dimensions: a list of dimensions to test
        :param block_sizes: a list of block sizes to test
        :param progressive_step_size: step size for the progressive strategy, or ``None`` to disable
            it
        :param log_filename: log to this file if not ``None``
        """

        self.classes = tuple(classes)
        self.matrixf = matrixf
        self.dimensions = tuple(dimensions)
        self.block_sizes = tuple(block_sizes)
        self.progressive_step_size = progressive_step_size
        self.log_filename = log_filename

    def __call__(self, seed, threads=2, samples=2, tours=1):
        """

        :param seed: A random seed, each matrix will be created with seed increased by one
        :param threads: number of threads to use
        :param samples: number of reductions to perform
        :param tours: number of BKZ tours to run

        """

        logger = logging.getLogger("compare")

        results = OrderedDict()

        fmtstring = "  %%%ds"%max([len(BKZ_.__name__) for BKZ_ in self.classes])

        for dimension in self.dimensions:
            results[dimension] = OrderedDict()

            for block_size in self.block_sizes:

                seed_ = seed

                if dimension < block_size:
                    continue

                L = OrderedDict([(BKZ_.__name__, OrderedDict()) for BKZ_ in self.classes])

                logger.info("dimension: %3d, block_size: %2d"%(dimension, block_size))

                tasks = []

                matrixf = self.matrixf(dimension=dimension, block_size=block_size)

                for i in range(samples):
                    set_random_seed(seed_)
                    A = IntegerMatrix.random(dimension, **matrixf)

                    for BKZ_ in self.classes:
                        args = (BKZ_, A, block_size, tours, self.progressive_step_size)
                        tasks.append(((seed_, BKZ_), args))

                    seed_ += 1

                if threads > 1:
                    pool = Pool(processes=threads)
                    tasks = dict([(key, pool.apply_async(bkz_call, args_)) for key, args_ in tasks])
                    pool.close()

                    while tasks:
                        ready = [key for key in tasks if tasks[key].ready()]
                        for key in ready:
                            seed_, BKZ_ = key
                            try:
                                trace_ = tasks[key].get()
                                L[BKZ_.__name__][seed_] = trace_
                                logger.debug(fmtstring%(BKZ_.__name__) +
                                             " 0x%08x %s"%(seed_, pretty_dict(trace_.data)))
                            except ReductionError:
                                logger.debug("ReductionError in %s with seed 0x%08x"%(BKZ_.__name__, seed_))
                            del tasks[key]

                        time.sleep(1)
                else:
                    for key, args_ in tasks:
                        seed_, BKZ_ = key
                        try:
                            trace_ = apply(bkz_call, args_)
                            L[BKZ_.__name__][seed_] = trace_
                            logger.debug(fmtstring%(BKZ_.__name__) +
                                         " 0x%08x %s"%(seed_, pretty_dict(trace_.data)))
                        except ReductionError:
                            logger.debug("ReductionError in %s with seed 0x%08x"%(BKZ_.__name__, seed_))

                logger.debug("")
                for name, vals in L.items():
                    if vals:
                        vals = OrderedDict(zip(vals.items()[0][1].data, zip(*[d[1].data.values() for d in vals.items()])))
                        vals = OrderedDict((k, float(sum(v))/len(v)) for k, v in vals.items())
                        logger.info(fmtstring%(name) + "    average %s"%(pretty_dict(vals)))

                logger.info("")
                results[dimension][block_size] = L

                self.write_log(results)

        return results

    def write_log(self, data):
        if self.log_filename is not None:
            pickle.dump(data, open(self.log_filename, "wb"))


# Example

class BKZGlue(object):
    "Base class for producing new BKZ classes with some parameters fixed."
    def tour(self, params, min_row=0, max_row=-1, tracer=dummy_tracer):
        if isinstance(params, int):
            params = BKZ.Param(block_size=params, **self.kwds)
        return self.base.tour(self, params, min_row=min_row, max_row=max_row, tracer=tracer)


def BKZFactory(name, BKZBase, **kwds):
    """
    Return a new BKZ class, derived from ``BKZBase`` with given ``name``.  The resulting class
    accepts a single ``block_size`` parameter for ``tour`` and substitutes it with a ``BKZ.Param```
    object where the keyword parameters provided to this function are fixed the values provided to
    this function ::

        >>> from fpylll.algorithms.bkz2 import BKZReduction as BKZ2
        >>> from fpylll import BKZ
        >>> from fpylll.tools.compare import BKZFactory
        >>> BKZ2_LOW = BKZFactory('BKZ2_LOW', BKZ2, strategies=BKZ.DEFAULT_STRATEGY, min_success_probability=0.1)

    :param name: name for output class
    :param BKZBase: base class to base this class on

    """
    NEW_BKZ = type(name, (BKZGlue, BKZBase), {"kwds": kwds, "base": BKZBase})
    globals()[name] = NEW_BKZ  # this is a HACK to enable pickling
    return NEW_BKZ


BKZ1 = BKZFactory("BKZ1", fpylll.algorithms.bkz.BKZReduction)
BKZ2 = BKZFactory("BKZ2", fpylll.algorithms.bkz2.BKZReduction, strategies=BKZ.DEFAULT_STRATEGY)
BKZ2r = BKZFactory("BKZ2r", fpylll.algorithms.bkz2.BKZReduction, strategies=BKZ.DEFAULT_STRATEGY, rerandomization_density=2)
BKZ2_otf = BKZFactory("BKZ2_otf", fpylll.algorithms.bkz2_otf.BKZReduction)

BKZ2_otf_subsol = BKZFactory("BKZ2_otf_subsol",
                             fpylll.algorithms.bkz2_otf_subsol.BKZReduction)

BKZ2_otf_ss_p1 = BKZFactory("BKZ2_otf_ss_p1",
                            fpylll.algorithms.bkz2_otf_subsol.BKZReduction,
                            min_success_probability=.5**1)

BKZ2_otf_ss_p2 = BKZFactory("BKZ2_otf_ss_p2",
                            fpylll.algorithms.bkz2_otf_subsol.BKZReduction,
                            min_success_probability=.5**2)

BKZ2_otf_ss_p3 = BKZFactory("BKZ2_otf_ss_p3",
                            fpylll.algorithms.bkz2_otf_subsol.BKZReduction,
                            min_success_probability=.5**3)

BKZ2_otf_ss_p4 = BKZFactory("BKZ2_otf_ss_p4",
                            fpylll.algorithms.bkz2_otf_subsol.BKZReduction,
                            min_success_probability=.5**4)

BKZ2_otf_ss_p5 = BKZFactory("BKZ2_otf_ss_p5",
                            fpylll.algorithms.bkz2_otf_subsol.BKZReduction,
                            min_success_probability=.5**5)

BKZ2_otf_ss_p6 = BKZFactory("BKZ2_otf_ss_p6",
                            fpylll.algorithms.bkz2_otf_subsol.BKZReduction,
                            min_success_probability=.5**6)

BKZ2_otf_ss_p7 = BKZFactory("BKZ2_otf_ss_p7",
                            fpylll.algorithms.bkz2_otf_subsol.BKZReduction,
                            min_success_probability=.5**7)



# Main


def qary30(dimension, block_size):
    return {"algorithm": "qary",
            "k": dimension//2,
            "bits": 30,
            "int_type": "long"}


def _setup_logging(verbose=False):
    import subprocess
    import datetime

    hostname = str(subprocess.check_output("hostname").rstrip())
    now = datetime.datetime.today().strftime("%Y-%m-%d-%H:%M")
    log_name = "compare-{hostname}-{now}".format(hostname=hostname, now=now)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(levelname)5s:%(name)s:%(asctime)s: %(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S %Z',
                        filename=log_name + ".log")

    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO if not verbose else logging.DEBUG)
    console.setFormatter(logging.Formatter('%(name)s: %(message)s',))
    logging.getLogger('compare').addHandler(console)

    return log_name


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='Measure Pressure')
    parser.add_argument('-c', '--classes', help='BKZ classes',
                        type=str, nargs='+', default=["BKZ2"])
    parser.add_argument('-f', '--files', help='additional files to load for BKZ classes',
                        type=str, nargs='+', default=list())
    parser.add_argument('-t', '--threads', help='number of threads to use', type=int, default=1)
    parser.add_argument('-r', '--tours',   help='number of BKZ tours', type=int, default=1)
    parser.add_argument('-s', '--samples', help='number of samples to try', type=int, default=4)
    parser.add_argument('-z', '--seed', help="random seed", type=int, default=0x1337)
    parser.add_argument('-b', '--block-sizes', help='block sizes',
                        type=int,  nargs='+', default=(10, 20, 30, 40))
    parser.add_argument('-p', '--progressive-step-size', help='step size for progressive strategy, None for disabled',
                        default=None, type=int)
    parser.add_argument('-d', '--dimensions', help='lattice dimensions',
                        type=int, nargs='+', default=(60, 80, 100, 120))
    parser.add_argument('-v', '--verbose', help='print more details by default', action='store_true')

    return parser.parse_args()


def _find_classes(class_names, filenames):
    import imp
    import os
    import re

    classes = class_names
    classes = [globals().get(clas, clas) for clas in classes]

    for i, fn in enumerate(filenames):
        tmp = imp.load_source("compare_module%03d"%i, fn)
        # find the class by name in the module
        classes = [tmp.__dict__.get(clas, clas) for clas in classes]

        # check if there's some BKZReduction implemented in bkz_foo.py, we match this with BKZ_FOO
        if fn.startswith(os.path.basename(fn)) and "BKZReduction" in tmp.__dict__:
            candidate = re.sub("bkz(.*)\.py", "BKZ\\1", os.path.basename(fn)).upper()
            if candidate in classes:
                tmp.BKZReduction.__name__ = candidate
                classes[classes.index(candidate)] = tmp.BKZReduction

    for clas in classes:
        if isinstance(clas, str):
            raise ValueError("Cannot find '%s'"%clas)

    return classes


if __name__ == '__main__':
    import pickle

    args = _parse_args()
    classes = _find_classes(args.classes, args.files)
    log_filename = _setup_logging(args.verbose)

    for k, v in sorted(vars(args).items()):
        logging.getLogger('compare').debug("%s: %s"%(k, v))

    compare_bkz = CompareBKZ(classes=classes,
                             matrixf=qary30,
                             block_sizes=args.block_sizes,
                             progressive_step_size=args.progressive_step_size,
                             dimensions=args.dimensions,
                             log_filename=log_filename + ".sobj")

    results = compare_bkz(seed=args.seed,
                          threads=args.threads,
                          samples=args.samples,
                          tours=args.tours)
