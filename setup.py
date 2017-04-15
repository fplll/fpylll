#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess
import sys

try:
    from itertools import ifilter as filter
except ImportError:
    pass  # python 3

from os import path
from ast import parse
from distutils.core import setup
from distutils.extension import Extension
import Cython.Build

from copy import copy

if "READTHEDOCS" in os.environ:
    # When building with readthedocs, install the dependencies too.
    # See https://github.com/rtfd/readthedocs.org/issues/2776
    for reqs in ["requirements.txt", "suggestions.txt"]:
        if os.path.isfile(reqs):
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", reqs])


# CONFIG VARIABLES

cythonize_dir = "build"

include_dirs = [os.path.join(sys.prefix, "include")]
library_dirs = [os.path.join(sys.exec_prefix, "lib")]

fplll = {"include_dirs": include_dirs,
         "library_dirs": library_dirs,
         "language": "c++",
         "libraries": ["gmp", "mpfr", "fplll"],
         "extra_compile_args": ["-std=c++11"],
         "extra_link_args": ["-std=c++11"]}

other = {"include_dirs": include_dirs,
         "library_dirs": library_dirs,
         "libraries": ["gmp"]}

if "READTHEDOCS" in os.environ:
    # ReadTheDocs uses fplll from Conda, which was built with the old
    # C++ ABI.
    fplll["extra_compile_args"].append("-D_GLIBCXX_USE_CXX11_ABI=0")

config_pxi = []


# QD
have_qd = False

try:
    libs = subprocess.check_output(["pkg-config", "fplll", "--libs"])
    if b"-lqd" in libs:
        have_qd = True
except subprocess.CalledProcessError:
    pass

if have_qd:
    fplll["libraries"].append("qd")
    config_pxi.append("DEF HAVE_QD=True")
else:
    config_pxi.append("DEF HAVE_QD=False")


# NUMPY

try:
    import numpy
    have_numpy = True
except ImportError:
    have_numpy = False

if have_numpy:
    config_pxi.append("DEF HAVE_NUMPY=True")
    numpy_args = copy(fplll)
    numpy_args["include_dirs"].append(numpy.get_include())
else:
    config_pxi.append("DEF HAVE_NUMPY=False")

# Ideally this would check the fplll headers explicitly for the
# the FPLLL_WITH_LONG_DOUBLE define, but for now it suffices to
# say that long double support is disabled on Cygwin
have_long_double = not sys.platform.startswith('cygwin')
config_pxi.append("DEF HAVE_LONG_DOUBLE={0}".format(have_long_double))


# CONFIG.PXI
config_pxi_path = os.path.join(".", "src", "fpylll", "config.pxi")
config_pxi = "\n".join(config_pxi) + "\n"

try:
    cur_config_pxi = open(config_pxi_path, "r").read()
except IOError:
    cur_config_pxi = ""

if cur_config_pxi != config_pxi:  # check if we need to write
    with open(config_pxi_path, "w") as fw:
        fw.write(config_pxi)


# EXTENSIONS

extensions = [
    Extension("fpylll.gmp.pylong", ["src/fpylll/gmp/pylong.pyx"], **other),
    Extension("fpylll.fplll.integer_matrix", ["src/fpylll/fplll/integer_matrix.pyx"], **fplll),
    Extension("fpylll.fplll.gso", ["src/fpylll/fplll/gso.pyx"], **fplll),
    Extension("fpylll.fplll.lll", ["src/fpylll/fplll/lll.pyx"], **fplll),
    Extension("fpylll.fplll.wrapper", ["src/fpylll/fplll/wrapper.pyx"], **fplll),
    Extension("fpylll.fplll.bkz_param", ["src/fpylll/fplll/bkz_param.pyx"], **fplll),
    Extension("fpylll.fplll.bkz", ["src/fpylll/fplll/bkz.pyx"], **fplll),
    Extension("fpylll.fplll.enumeration", ["src/fpylll/fplll/enumeration.pyx"], **fplll),
    Extension("fpylll.fplll.svpcvp", ["src/fpylll/fplll/svpcvp.pyx"], **fplll),
    Extension("fpylll.fplll.pruner", ["src/fpylll/fplll/pruner.pyx"], **fplll),
    Extension("fpylll.util", ["src/fpylll/util.pyx"], **fplll),
    Extension("fpylll.io", ["src/fpylll/io.pyx"], **fplll),
    Extension("fpylll.config", ["src/fpylll/config.pyx"], **fplll),
]

if have_numpy:
    extensions.append(Extension("fpylll.numpy", ["src/fpylll/numpy.pyx"], **numpy_args))


# VERSION

with open(path.join('src', 'fpylll', '__init__.py')) as f:
    __version__ = parse(next(filter(lambda line: line.startswith('__version__'), f))).body[0].value.s


# FIRE

setup(
    name="fpylll",
    description="A Python interface for https://github.com/fplll/fplll",
    author=u"Martin R. Albrecht",
    author_email="fplll-devel@googlegroups.com",
    url="https://github.com/fplll/fpylll",
    version=__version__,
    ext_modules=Cython.Build.cythonize(extensions,
                                       include_path=["src"],
                                       build_dir=cythonize_dir,
                                       compiler_directives={'binding': True, "embedsignature": True}),
    package_dir={"": "src"},
    packages=["fpylll", "fpylll.gmp", "fpylll.fplll", "fpylll.algorithms", "fpylll.tools"],
    license='GNU General Public License, version 2 or later',
    long_description=open('README.rst').read(),
)
