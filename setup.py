#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import platform
import subprocess
import sys
import io

if "READTHEDOCS" in os.environ:
    # When building with readthedocs, install the dependencies too.
    # See https://github.com/rtfd/readthedocs.org/issues/2776
    for reqs in ["requirements.txt", "suggestions.txt"]:
        if os.path.isfile(reqs):
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", reqs])

try:
    from itertools import ifilter as filter
except ImportError:
    pass  # python 3

from os import path
from ast import parse

try:
    from setuptools.command.build_ext import build_ext as _build_ext
    from setuptools.core import setup
    from setuptools.extension import Extension as _Extension
    aux_setup_kwds = {"install_requires": ["cysignals"]}
except ImportError:
    from distutils.command.build_ext import build_ext as _build_ext
    from distutils.core import setup
    from distutils.extension import Extension as _Extension
    aux_setup_kwds = {}

from copy import copy

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = OSError  # Python 2 workaround


class Extension(_Extension, object):
    """
    distutils.extension.Extension subclass supporting additional
    keywords:

        * fplll: compile and link with flags from fplll as defined below in
          build_ext.fplll below
        * other: flags for compiling and linking other extension modules
          (without fplll flags) as defined below in build_ext.other
    """

    def __init__(self, name, sources, **kwargs):
        self.fplll = kwargs.pop("fplll", False)
        self.other = kwargs.pop("other", False)
        super(Extension, self).__init__(name, sources, **kwargs)


class build_ext(_build_ext, object):
    # CONFIG VARIABLES

    cythonize_dir = "build"
    fplll = None
    other = None
    def_varnames = ["HAVE_QD", "HAVE_LONG_DOUBLE", "HAVE_NUMPY"]
    config_pxi_path = os.path.join(".", "src", "fpylll", "config.pxi")

    def finalize_options(self):
        super(build_ext, self).finalize_options()

        def_vars = self._generate_config_pxi()

        include_dirs = [os.path.join(sys.prefix, "include")]
        library_dirs = [os.path.join(sys.exec_prefix, "lib")]
        cxxflags = list(filter(None, os.environ.get("CXXFLAGS", "").split()))

        if self.fplll is None:
            self.fplll = {
                "include_dirs": include_dirs,
                "library_dirs": library_dirs,
                "language": "c++",
                "libraries": ["gmp", "mpfr", "fplll"],
                "extra_compile_args": ["-std=c++11"] + cxxflags,
                "extra_link_args": ["-std=c++11"],
                "define_macros": [("__PYX_EXTERN_C", 'extern "C++"')],
            }

            if def_vars["HAVE_QD"]:
                self.fplll["libraries"].append("qd")

        if self.other is None:
            self.other = {
                "include_dirs": include_dirs,
                "library_dirs": library_dirs,
                "libraries": ["gmp"],
            }

        if "READTHEDOCS" in os.environ:
            # ReadTheDocs uses fplll from Conda, which was built with the old
            # C++ ABI.
            self.fplll["extra_compile_args"].append("-D_GLIBCXX_USE_CXX11_ABI=0")

        if def_vars["HAVE_NUMPY"]:
            import numpy

            numpy_args = copy(self.fplll)
            numpy_args["include_dirs"].append(numpy.get_include())
            self.extensions.append(
                Extension("fpylll.numpy", ["src/fpylll/numpy.pyx"], **numpy_args)
            )

        for ext in self.extensions:
            if ext.fplll:
                for key, value in self.fplll.items():
                    setattr(ext, key, value)
            elif ext.other:
                for key, value in self.other.items():
                    setattr(ext, key, value)

    def run(self):
        import Cython.Build
        self.extensions = Cython.Build.cythonize(
            self.extensions,
            include_path=["src"],
            build_dir=self.cythonize_dir,
            compiler_directives={"binding": True, "embedsignature": True, "language_level": 2},
        )
        super(build_ext, self).run()

    def _generate_config_pxi(self):
        def_vars = {}
        config_pxi = []

        for defvar in self.def_varnames:
            # We can optionally read values for these variables for the
            # environment; this is mostly used to force different values for
            # testing
            value = os.environ.get(defvar)
            if value is not None:
                value = value.lower() in ["1", "true", "yes"]
            else:
                value = getattr(self, "_get_" + defvar.lower())()

            config_pxi.append("DEF {0}={1}".format(defvar, value))
            def_vars[defvar] = value

        config_pxi = "\n".join(config_pxi) + "\n"

        try:
            cur_config_pxi = open(self.config_pxi_path, "r").read()
        except IOError:
            cur_config_pxi = ""

        if cur_config_pxi != config_pxi:  # check if we need to write
            with open(self.config_pxi_path, "w") as fw:
                fw.write(config_pxi)

        return def_vars

    def _get_have_qd(self):
        if "CONDA_PREFIX" in os.environ:
            os.environ["PKG_CONFIG_PATH"] = ":".join(
                [
                    os.path.join(os.environ["CONDA_PREFIX"], "lib", "pkgconfig"),
                    os.environ.get("PKG_CONFIG_PATH", ""),
                ]
            )
        if "VIRTUAL_ENV" in os.environ:
            os.environ["PKG_CONFIG_PATH"] = ":".join(
                [
                    os.path.join(os.environ["VIRTUAL_ENV"], "lib", "pkgconfig"),
                    os.environ.get("PKG_CONFIG_PATH", ""),
                ]
            )
        try:
            libs = subprocess.check_output(["pkg-config", "fplll", "--libs"])
            if b"-lqd" in libs:
                return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        return False

    def _get_have_numpy(self):
        try:
            import numpy

            return True
        except ImportError:
            pass

        return False

    def _get_have_long_double(self):
        # Ideally this would check the fplll headers explicitly for the
        # the FPLLL_WITH_LONG_DOUBLE define, but for now it suffices to
        # say that long double support is disabled on Cygwin
        return not (sys.platform.startswith("cygwin") or ("macOS" in (_ := platform.platform()) and "arm" in _))


# EXTENSIONS

extensions = [
    Extension("fpylll.gmp.pylong", ["src/fpylll/gmp/pylong.pyx"], other=True),
    Extension("fpylll.fplll.integer_matrix", ["src/fpylll/fplll/integer_matrix.pyx"], fplll=True),
    Extension("fpylll.fplll.gso", ["src/fpylll/fplll/gso.pyx"], fplll=True),
    Extension("fpylll.fplll.lll", ["src/fpylll/fplll/lll.pyx"], fplll=True),
    Extension("fpylll.fplll.wrapper", ["src/fpylll/fplll/wrapper.pyx"], fplll=True),
    Extension("fpylll.fplll.bkz_param", ["src/fpylll/fplll/bkz_param.pyx"], fplll=True),
    Extension("fpylll.fplll.bkz", ["src/fpylll/fplll/bkz.pyx"], fplll=True),
    Extension("fpylll.fplll.enumeration", ["src/fpylll/fplll/enumeration.pyx"], fplll=True),
    Extension("fpylll.fplll.svpcvp", ["src/fpylll/fplll/svpcvp.pyx"], fplll=True),
    Extension("fpylll.fplll.pruner", ["src/fpylll/fplll/pruner.pyx"], fplll=True),
    Extension("fpylll.util", ["src/fpylll/util.pyx"], fplll=True),
    Extension("fpylll.io", ["src/fpylll/io.pyx"], fplll=True),
    Extension("fpylll.config", ["src/fpylll/config.pyx"], fplll=True),
]


# VERSION

with open(path.join("src", "fpylll", "__init__.py")) as f:
    __version__ = (
        parse(next(filter(lambda line: line.startswith("__version__"), f))).body[0].value.s
    )
# FIRE


def readme_to_long_description():
    """
    Python wants long descriptions to be plain ASCII.  Our contributors have names that are not
    plain ASCII. Thus, we cut off the list of contributors when reading the long description.
    """
    long_description = io.open("README.rst", encoding="utf-8").read()
    cut = long_description.index("Attribution & License")
    return str(long_description[:cut])


setup(
    name="fpylll",
    description="A Python interface for https://github.com/fplll/fplll",
    author=u"Martin R. Albrecht",
    author_email="fplll-devel@googlegroups.com",
    url="https://github.com/fplll/fpylll",
    version=__version__,
    ext_modules=extensions,
    package_dir={"": "src"},
    packages=["fpylll", "fpylll.gmp", "fpylll.fplll", "fpylll.algorithms", "fpylll.tools"],
    license="GNU General Public License, version 2 or later",
    long_description=readme_to_long_description(),
    cmdclass={"build_ext": build_ext},
    **aux_setup_kwds
)
