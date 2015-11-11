# -*- coding: utf-8 -*-
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import os

# TODO: Don't hardcode use of virtualenv

fplll = {"include_dirs": None,
         "library_dirs": None}

interrupt_include = os.path.join(".", "fpylll")

if "VIRTUAL_ENV" in os.environ:

    prefix = os.environ["VIRTUAL_ENV"]
    fplll["include_dirs"] = [os.path.join(prefix, "include"),  interrupt_include]
    fplll["library_dirs"] = [os.path.join(prefix, "lib")]

extensions = [
    Extension("interrupt.interrupt", ["fpylll/interrupt/interrupt.pyx"],
              include_dirs=[interrupt_include]),
    Extension("gmp.pylong", ["fpylll/gmp/pylong.pyx"]),
    Extension("util", ["fpylll/util.pyx"], **fplll),
    Extension("integer_matrix", ["fpylll/integer_matrix.pyx"], **fplll),
    Extension("gso", ["fpylll/gso.pyx"], **fplll),
    Extension("lll", ["fpylll/lll.pyx"], **fplll),
    Extension("wrapper", ["fpylll/wrapper.pyx"], **fplll),
    Extension("bkz", ["fpylll/bkz.pyx"], **fplll),
    Extension("enumeration", ["fpylll/enumeration.pyx"], **fplll),
    Extension("fpylll", ["fpylll/fpylll.pyx"], **fplll),
]

setup(
    name="fpyLLL",
    version='0.1dev',
    ext_package='fpylll',
    ext_modules=cythonize(extensions),
    packages=["fpylll", "fpylll.gmp", "fpylll.interrupt", "fpylll.contrib"],
    license='GNU General Public License, version 2 or later',
    long_description=open('README.rst').read(),
)
