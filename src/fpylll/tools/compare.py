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
from fpylll import FPLLL
from fpylll.tools.bkz_stats import BKZTreeTracer, dummy_tracer, pretty_dict
from fpylll.tools.quality import basis_quality
from fpylll.util import ReductionError

from multiprocessing import Pool
import logging
import copy
import time
import pickle

import fpylll.algorithms.bkz
import fpylll.algorithms.bkz2


# Utility Functions

def play(BKZ, A, block_size, tours, progressive_step_size=None):
    """Call ``BKZ`` on ``A`` with ``block_size`` for the given number of ``tours``.

    The given number of tours is used for all block sizes from 2 up to ``block_size`` in increments of
    ``progressive_step_size``. Providing ``None`` for this parameter disables the progressive strategy.

    :param BKZ: a BKZ class whose ``__call__`` accepts a single block size as parameter
    :param A: an integer matrix
    :param block_size: a block size >= 2
    :param tours: number of tours >= 1
    :param progressive_step_size: step size for progressive strategy
    :returns: a trace of the execution using ``BKZTreeTracer``

    ..  note :: This function essentially reimplements ``BKZ.__call__`` but supports the
        progressive strategy.
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


class Conductor(object):
    """
    A conductor is our main class for launching block-wise lattice reductions and collecting the outputs.
    """
    def __init__(self, threads=1, pickle_jar=None, logger="."):
        """Create a new conductor object.

        :param threads: number of threads
        :param pickle_jar: dump traces to this file continuously

        """
        self.pool = Pool(processes=threads)
        self.threads = threads
        self.pickle_jar = pickle_jar
        self.logger = logging.getLogger(logger)
        self.outputs = OrderedDict()
        self._major_strlen = 0
        self._minor_strlen = 0

    def _majorminor_format_str(self):
        "Used to align log files"
        return "%%%ds(%%%ds) :: %%s"%(self._major_strlen, self._minor_strlen)

    def _update_strlens(self, major, minor):
        "Update string lengths of major/minor tags"
        self._major_strlen = max(len(str(major)), self._major_strlen)
        self._minor_strlen = max(len(str(minor)), self._minor_strlen)

    @staticmethod
    def dump(data, filename):
        "Pickle ``data`` to ``filename``"
        pickle.dump(data, open(filename, "wb"))

    def wait_on(self, outputs, todo, sleep=1):
        """Wait for jobs in ``todo`` to return and store results in ``outputs``.

        :param outputs: store results here
        :param todo: these are running jobs
        :param sleep: seconds to sleep before checking if new results are availabl.

        """

        fmtstr = self._majorminor_format_str()

        while todo:
            collect = [(tag, res) for (tag, res) in todo if res.ready()]

            for tag, res in collect:
                major, minor = tag
                try:
                    res = res.get()
                    if major not in outputs:
                        outputs[major] = []
                    outputs[major].append((minor, res))
                    self.logger.debug(fmtstr%(major, minor, pretty_dict(res.data)))

                    if self.pickle_jar is not None:
                        Conductor.dump(self.outputs, self.pickle_jar)

                except ReductionError:
                    self.logger.debug("ReductionError for %s(%s)."%(major, minor))

            todo = todo.difference(collect)
            time.sleep(sleep)

        return outputs

    def log_averages(self, tags, outputs):
        """
        Log average values for all entries tagged as ``tags`` in ``outputs``.
        """
        fmtstr = self._majorminor_format_str()
        avg = OrderedDict()

        for major, minor in tags:
            if major in avg:
                continue
            avg[major] = OrderedDict()
            n = len(outputs[major])
            for minor, output in outputs[major]:
                for k, v in output.data.items():
                    avg[major][k] =  avg[major].get(k, 0.0) + float(v)/n

            self.logger.info(fmtstr%(major, "avg", pretty_dict(avg[major])))

    def __call__(self, jobs, current=None):
        """
        Call ``jobs`` in parallel.

        The parameter jobs is a list with the following format.  Each entry is one of the following:

            - a tuple ``((major, minor), (BKZ, A, block_size, tours, progressive_step_size))``,
              where ``major`` and ``minor`` are arbitrary hashable tags and the rest are valid
              inputs to ``play``.

            - A list with elements of the same format as above.

        Entries at the same level are considered to be a group.  All jobs in the same group go into
        the same execution pool.  At the end of the execution of a group the average across all
        ``minor`` tags of a ``major`` tag are shown.

        ..  note :: Recursive jobs, i.e. those in a sub-list are run first, this is an
            implementation artefact.  Typically, we don't expect jobs and lists of jobs to be mixed at
            the same level, though, this is supported.

        """
        inputs = OrderedDict()
        if current is None:
            current = self.outputs

        # filter out sub-jobs that should be grouped and call recursively
        for tag, job in jobs:
            if isinstance(job[0], (list, tuple)):
                self.logger.info("")
                self.logger.info("# %s (size: %d) #"%(tag, len(job)))
                current[tag] = OrderedDict()
                self(job, current=current[tag])
            else:
                major, minor = tag
                self._update_strlens(major, minor)
                if major not in current:
                    current[major] = list()
                inputs[tag] = job

        self.logger.debug("")

        # base case
        if self.threads > 1:
            todo = set()
            for tag in inputs:
                todo.add((tag, self.pool.apply_async(play, inputs[tag])))

            current = self.wait_on(current, todo)

        else:
            fmtstr = self._majorminor_format_str()
            for tag in inputs:
                major, minor = tag
                try:
                    res = play(*inputs[tag])
                    current[major].append((minor, res))
                    self.logger.debug(fmtstr%(major, minor, pretty_dict(res.data)))

                    if self.pickle_jar is not None:
                        Conductor.dump(self.outputs, self.pickle_jar)

                except ReductionError:
                    self.logger.debug("ReductionError for %s(%s)."%(major, minor))

        self.logger.debug("")

        # print averages per major tag
        self.log_averages(inputs.keys(), current)

        if self.pickle_jar is not None:
            Conductor.dump(self.outputs, self.pickle_jar)

        return self.outputs


def compare_bkz(classes, matrixf, dimensions, block_sizes, progressive_step_size,
                seed, threads=2, samples=2, tours=1,
                pickle_jar=None, logger="compare"):
    """
    Compare BKZ-style lattice reduction.

    :param classes: a list of BKZ classes to test.  See caveat above.
    :param matrixf: A function to create matrices for a given dimension and block size
    :param dimensions: a list of dimensions to test
    :param block_sizes: a list of block sizes to test
    :param progressive_step_size: step size for the progressive strategy; ``None`` to disable it
    :param seed: A random seed, each matrix will be created with seed increased by one
    :param threads: number of threads to use
    :param samples: number of reductions to perform
    :param tours: number of BKZ tours to run
    :param log_filename: log to this file if not ``None``

    """

    jobs = []

    for dimension in dimensions:

        jobs.append((dimension, []))

        for block_size in block_sizes:
            if dimension < block_size:
                continue

            seed_ = seed
            jobs_ = []

            matrixf_ = matrixf(dimension=dimension, block_size=block_size)

            for i in range(samples):
                FPLLL.set_random_seed(seed_)
                A = IntegerMatrix.random(dimension, **matrixf_)

                for BKZ_ in classes:
                    args = (BKZ_, A, block_size, tours, progressive_step_size)
                    jobs_.append(((BKZ_.__name__, seed_), args))
                seed_ += 1

            jobs[-1][1].append((block_size, jobs_))

    conductor = Conductor(threads=threads, pickle_jar=pickle_jar, logger=logger)
    return conductor(jobs)


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


# Main

def qary30(dimension, block_size):
    return {"algorithm": "qary",
            "k": dimension//2,
            "bits": 30,
            "int_type": "long"}


def setup_logging(name, verbose=False):
    import subprocess
    import datetime

    hostname = str(subprocess.check_output("hostname").rstrip())
    now = datetime.datetime.today().strftime("%Y-%m-%d-%H:%M")
    log_name = "{name}-{hostname}-{now}".format(name=name, hostname=hostname, now=now)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(levelname)5s:%(name)s:%(asctime)s: %(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S %Z',
                        filename=log_name + ".log")

    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO if not verbose else logging.DEBUG)
    console.setFormatter(logging.Formatter('%(name)s: %(message)s',))
    logging.getLogger(name).addHandler(console)

    return log_name


def names_to_classes(class_names, filenames):
    """
    Try to find a class for each name in ``class_names``.  Classes implemented in one of the
    ``filenames`` are also considered.

    The following mapping logic is used:

        - if a class with ``name`` exists, it is used

        - if ``name`` is ``BKZ_FOO`` and a class called ``BKZReduction`` is implemented in a file
          ``bkz_foo`` it is used.

    :param class_names:
    :param filenames:

    """
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
            candidate = re.sub("bkz(.*)\\.py", "BKZ\\1", os.path.basename(fn)).upper()
            if candidate in classes:
                tmp.BKZReduction.__name__ = candidate
                classes[classes.index(candidate)] = tmp.BKZReduction

    for clas in classes:
        if isinstance(clas, str):
            raise ValueError("Cannot find '%s'"%clas)

    return classes


if __name__ == '__main__':
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

    args =  parser.parse_args()

    name = "compare"

    classes = names_to_classes(args.classes, args.files)
    log_filename = setup_logging(name, args.verbose)

    for k, v in sorted(vars(args).items()):
        logging.getLogger(name).debug("%s: %s"%(k, v))

    results = compare_bkz(classes=classes,
                          matrixf=qary30,
                          block_sizes=args.block_sizes,
                          progressive_step_size=args.progressive_step_size,
                          dimensions=args.dimensions,
                          logger=name,
                          pickle_jar=log_filename + ".sobj",
                          seed=args.seed,
                          threads=args.threads,
                          samples=args.samples,
                          tours=args.tours)
