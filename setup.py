#!/usr/bin/env python
# -*- coding: utf-8 -*-
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import sys
import os

cythonize_dir = "build"

fplll = {"include_dirs": None,
         "library_dirs": None}

if "VIRTUAL_ENV" in os.environ:
    prefix = os.environ["VIRTUAL_ENV"]
    fplll["include_dirs"] = [os.path.join(prefix, "include")]
    fplll["library_dirs"] = [os.path.join(prefix, "lib")]

extensions = [
    Extension("gmp.pylong", ["src/fpylll/gmp/pylong.pyx"], **fplll),
    Extension("util", ["src/fpylll/util.pyx"], **fplll),
    Extension("integer_matrix", ["src/fpylll/integer_matrix.pyx"], **fplll),
    Extension("gso", ["src/fpylll/gso.pyx"], **fplll),
    Extension("lll", ["src/fpylll/lll.pyx"], **fplll),
    Extension("wrapper", ["src/fpylll/wrapper.pyx"], **fplll),
    Extension("bkz", ["src/fpylll/bkz.pyx"], **fplll),
    Extension("enumeration", ["src/fpylll/enumeration.pyx"], **fplll),
    Extension("svpcvp", ["src/fpylll/svpcvp.pyx"], **fplll),
    Extension("fpylll", ["src/fpylll/fpylll.pyx"], **fplll),
]

setup(
    name="fpyLLL",
    version='0.1dev',
    ext_package='fpylll',
    ext_modules=cythonize(extensions,
                          include_path=["src"] + sys.path,
                          build_dir=cythonize_dir),
    package_dir={"": "src"},
    packages=["fpylll", "fpylll.gmp", "fpylll.contrib"],
    license='GNU General Public License, version 2 or later',
    long_description=open('README.rst').read(),
)
