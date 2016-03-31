#!/usr/bin/env python
# -*- coding: utf-8 -*-
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import os
import subprocess
import sys
from copy import copy


# CONFIG VARIABLES

cythonize_dir = "build"

fplll = {"include_dirs": [],
         "library_dirs": [],
         "language": "c++",
         "libraries": ["gmp", "mpfr", "fplll"]}

other = {"include_dirs": [],
         "library_dirs": [],
         "libraries": ["gmp"]}

config_pxi = []


# VIRTUALENVS

if "VIRTUAL_ENV" in os.environ:
    prefix = os.environ["VIRTUAL_ENV"]
    fplll["include_dirs"] = [os.path.join(prefix, "include")]
    fplll["library_dirs"] = [os.path.join(prefix, "lib")]
    other["include_dirs"] = [os.path.join(prefix, "include")]
    other["library_dirs"] = [os.path.join(prefix, "lib")]


# QD
have_qd = False

try:
    libs = subprocess.check_output(["pkg-config", "fplll", "--libs"])
    if "-lqd" in libs:
        have_qd = True
except subprocess.CalledProcessError:
    pass

if have_qd:
    fplll["libraries"].append("qd")
    config_pxi.append("DEF HAVE_QD=True")
else:
    config_pxi.append("DEF HAVE_QD=False")


# SAGE
have_sage = False

try:
    import sage
    have_sage = True
except ImportError:
    pass

if have_sage:
    fplll["include_dirs"].append(os.getenv("SAGE_SRC"))
    config_pxi.append("DEF HAVE_SAGE=True")
else:
    config_pxi.append("DEF HAVE_SAGE=False")

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
    Extension("gmp.pylong", ["src/fpylll/gmp/pylong.pyx"], **other),
    Extension("util", ["src/fpylll/util.pyx"], **fplll),
    Extension("io", ["src/fpylll/io.pyx"], **fplll),
    Extension("integer_matrix", ["src/fpylll/integer_matrix.pyx"], **fplll),
    Extension("gso", ["src/fpylll/gso.pyx"], **fplll),
    Extension("lll", ["src/fpylll/lll.pyx"], **fplll),
    Extension("wrapper", ["src/fpylll/wrapper.pyx"], **fplll),
    Extension("bkz", ["src/fpylll/bkz.pyx"], **fplll),
    Extension("enumeration", ["src/fpylll/enumeration.pyx"], **fplll),
    Extension("svpcvp", ["src/fpylll/svpcvp.pyx"], **fplll),
    Extension("fpylll", ["src/fpylll/fpylll.pyx"], **fplll),
]

if have_numpy:
    extensions.append(Extension("numpy", ["src/fpylll/numpy.pyx"], **numpy_args))

setup(
    name="fpyLLL",
    version='0.1dev',
    ext_package='fpylll',
    ext_modules=cythonize(extensions,
                          include_path=["src"] + sys.path,
                          build_dir=cythonize_dir,
                          compiler_directives={'embedsignature': True}),
    package_dir={"": "src"},
    packages=["fpylll", "fpylll.gmp", "fpylll.contrib"],
    license='GNU General Public License, version 2 or later',
    long_description=open('README.rst').read(),
)
