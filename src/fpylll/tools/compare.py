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
from multiprocessing import Queue, Process

import logging
import copy
import fpylll.algorithms.bkz
import fpylll.algorithms.bkz2


# Utility Functions

def chunk_iterator(lst, step):
    """Return up to ``step`` entries from ``lst`` each time this function is called.

    :param lst: a list
    :param step: number of elements to return

    """
    for i in range(0, len(lst), step):
        yield tuple(lst[j] for j in range(i, min(i+step, len(lst))))


def bkz_call(BKZ, A, block_size, tours, return_queue=None, tag=None):
    """Call ``BKZ`` on ``A`` with ``block_size`` for the given number of ``tours``.

    If ``return_queue`` is not ``None`` then the trace and the provided ``tag`` are put on the
    queue.  Otherwise, they are returned.

    :param BKZ:
    :param A:
    :param block_size:
    :param tours:
    :param return_queue:
    :param tag:

    """
    bkz = BKZ(copy.copy(A))
    tracer = BKZTreeTracer(bkz, start_clocks=True)

    # this essentially initialises the GSO object, LLL was already run by the constructor, so this
    # is quick.
    with tracer.context("lll"):
        bkz.lll_obj()

    for i in range(tours):
        with tracer.context("tour", i):
            bkz.tour(block_size, tracer=tracer)

    tracer.exit()
    trace = tracer.trace

    quality = basis_quality(bkz.M)
    for k, v in quality.items():
        trace.data[k] = v

    if return_queue:
        return_queue.put((tag, trace))
    else:
        return (tag, trace)


class CompareBKZ:
    def __init__(self, classes, matrixf, dimensions, block_sizes):
        """

        :param classes: a list of BKZ classes to test. See caveat above.
        :param matrixf: A function to create matrices for a given dimension and block size
        :param dimensions: a list of dimensions to test
        :param block_sizes: a list of block sizes to test

        """

        self.classes = tuple(classes)
        self.matrixf = matrixf
        self.dimensions = tuple(dimensions)
        self.block_sizes = tuple(block_sizes)

    def __call__(self, seed, threads=2, samples=2, tours=1):
        """

        :param seed: A random seed, each matrix will be created with seed increased by one
        :param threads: number of threads to use
        :param samples: number of reductions to perform
        :param tours: number of BKZ tours to run

        """

        results = OrderedDict()

        for dimension in self.dimensions:
            results[dimension] = OrderedDict()
            for block_size in self.block_sizes:
                if dimension < block_size:
                    continue

                results[dimension][block_size] = OrderedDict()
                L = results[dimension][block_size]
                logging.info("dimension: %3d, block_size: %2d"%(dimension, block_size))

                tasks = []
                return_queue = Queue()

                matrixf = self.matrixf(dimension=dimension, block_size=block_size)

                for i in range(samples):
                    set_random_seed(seed)
                    A = IntegerMatrix.random(dimension, **matrixf)

                    for BKZ_ in self.classes:
                        L[BKZ_.__name__] = L.get(BKZ_.__name__, [])
                        args = (BKZ_, A, block_size, tours, return_queue, (BKZ_, seed))
                        task = Process(target=bkz_call, args=args)
                        tasks.append((BKZ_, task, args, seed))

                    seed += 1

                for chunk in chunk_iterator(tasks, threads):
                    for BKZ_, task, args, seed_ in chunk:
                        if threads > 1:
                            task.start()
                        else:
                            bkz_call(*args)

                    for _ in chunk:
                        if threads > 1:
                            task.join()

                        (BKZ_, seed_), trace = return_queue.get()
                        L[BKZ_.__name__].append((seed, trace))
                        logging.info("  %16s 0x%08x %s"%(BKZ_.__name__[:16], seed_, pretty_dict(trace.data)))

                logging.info("")
                for name, vals in L.items():
                    vals = OrderedDict(zip(vals[0][1].data, zip(*[d[1].data.values() for d in vals])))
                    vals = OrderedDict((k, float(sum(v))/len(v)) for k, v in vals.items())
                    logging.info("  %16s    average %s"%(name[:16], pretty_dict(vals)))

                logging.info("")
        return results


# Example

class BKZ1(fpylll.algorithms.bkz.BKZReduction):
    def tour(self, params, min_row=0, max_row=-1, tracer=dummy_tracer):
        if isinstance(params, int):
            params = BKZ.Param(block_size=params)
        return fpylll.algorithms.bkz.BKZReduction.tour(self, params, tracer=tracer)


class BKZ2(fpylll.algorithms.bkz2.BKZReduction):
    def tour(self, params, min_row=0, max_row=-1, tracer=dummy_tracer):
        if isinstance(params, int):
            params = BKZ.Param(block_size=params,
                               strategies=BKZ.DEFAULT_STRATEGY)
        return fpylll.algorithms.bkz2.BKZReduction.tour(self, params, tracer=tracer)


# Main

def qary30(dimension, block_size):
    return {"algorithm": "qary",
            "k": dimension//2,
            "bits": 30,
            "int_type": "long"}


def _setup_logging():
    import subprocess
    import datetime

    hostname = str(subprocess.check_output("hostname").rstrip())
    now = datetime.datetime.today().strftime("%Y-%m-%d-%H:%M")
    log_name = "compare-{hostname}-{now}".format(hostname=hostname, now=now)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(levelname)s:%(name)s:%(asctime)s: %(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S %Z',
                        filename=log_name + ".log")

    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(name)s: %(message)s',))
    logging.getLogger('').addHandler(console)

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
    parser.add_argument('-d', '--dimensions', help='lattice dimensions',
                        type=int, nargs='+', default=(60, 80, 100, 120))

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
    log_name = _setup_logging()

    for k, v in sorted(vars(args).items()):
        logging.debug("%s: %s"%(k, v))

    compare_bkz = CompareBKZ(classes=classes,
                             matrixf=qary30,
                             block_sizes=args.block_sizes,
                             dimensions=args.dimensions)

    results = compare_bkz(seed=args.seed,
                          threads=args.threads,
                          samples=args.samples,
                          tours=args.tours)
    pickle.dump(results, open(log_name + ".sobj", "wb"))
