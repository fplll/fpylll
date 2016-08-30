#!/usr/bin/env python
# -*- coding: utf-8 -*-
from distutils.core import setup
from distutils.extension import Extension
import Cython.Build

import os
import subprocess
import sys
from copy import copy


# CONFIG VARIABLES

cythonize_dir = "build"

fplll = {"include_dirs": [],
         "library_dirs": [],
         "language": "c++",
         "libraries": ["gmp", "mpfr", "fplll"],
         "extra_compile_args": ["-std=c++11"],
         "extra_link_args": ["-std=c++11"]}

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
    if b"-lqd" in libs:
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
    Extension("fplll.integer_matrix", ["src/fpylll/fplll/integer_matrix.pyx"], **fplll),
    Extension("fplll.gso", ["src/fpylll/fplll/gso.pyx"], **fplll),
    Extension("fplll.lll", ["src/fpylll/fplll/lll.pyx"], **fplll),
    Extension("fplll.wrapper", ["src/fpylll/fplll/wrapper.pyx"], **fplll),
    Extension("fplll.bkz_param", ["src/fpylll/fplll/bkz_param.pyx"], **fplll),
    Extension("fplll.bkz", ["src/fpylll/fplll/bkz.pyx"], **fplll),
    Extension("fplll.enumeration", ["src/fpylll/fplll/enumeration.pyx"], **fplll),
    Extension("fplll.svpcvp", ["src/fpylll/fplll/svpcvp.pyx"], **fplll),
    Extension("fplll.pruner", ["src/fpylll/fplll/pruner.pyx"], **fplll),
    Extension("util", ["src/fpylll/util.pyx"], **fplll),
    Extension("io", ["src/fpylll/io.pyx"], **fplll),
    Extension("config", ["src/fpylll/config.pyx"], **fplll),
]

if have_numpy:
    extensions.append(Extension("numpy", ["src/fpylll/numpy.pyx"], **numpy_args))


setup(
    setup_requires=[
        'cython>=0.x',
    ],
    name="fpylll",
    author=u"Martin R. Albrecht",
    author_email="fplll-devel@googlegroups.com",
    url="https://github.com/fplll/fpylll",
    version='0.2dev',
    ext_package='fpylll',
    ext_modules=Cython.Build.cythonize(extensions,
                                       include_path=["src"] + sys.path,
                                       build_dir=cythonize_dir,
                                       compiler_directives={'embedsignature': True}),
    package_dir={"": "src"},
    packages=["fpylll", "fpylll.gmp", "fpylll.fplll", "fpylll.algorithms", "fpylll.tools"],
    license='GNU General Public License, version 2 or later',
    long_description=open('README.rst').read(),
)
