# -*- coding: utf-8 -*-
include "fpylll/config.pxi"


from .fplll.fplll cimport default_strategy as default_strategy_c
from .fplll.fplll cimport default_strategy_path as default_strategy_path_c

from .fplll.fplll cimport FPLLL_MAJOR_VERSION as FPLLL_MAJOR_VERSION_c
from .fplll.fplll cimport FPLLL_MINOR_VERSION as FPLLL_MINOR_VERSION_c
from .fplll.fplll cimport FPLLL_MICRO_VERSION as FPLLL_MICRO_VERSION_c
from .fplll.fplll cimport FPLLL_MAX_ENUM_DIM as FPLLL_MAX_ENUM_DIM_c
from .fplll.fplll cimport FPLLL_HAVE_RECURSIVE_ENUM as FPLLL_HAVE_RECURSIVE_ENUM_c
from .fplll.fplll cimport FPLLL_MAX_PARALLEL_ENUM_DIM as FPLLL_MAX_PARALLEL_ENUM_DIM_c

IF HAVE_LONG_DOUBLE:
    have_long_double = True
    float_types = ("d", "ld")
ELSE:
    have_long_double = False
    float_types = ("d",)

IF HAVE_QD:
    have_qd = True
    float_types = float_types + ("dpe", "dd", "qd", "mpfr")
ELSE:
    have_qd = False
    float_types = float_types + ("dpe", "mpfr")

int_types             = ("long", "mpz")

default_strategy      = default_strategy_c().c_str()
default_strategy_path = default_strategy_path_c().c_str()

major_version         = FPLLL_MAJOR_VERSION_c
minor_version         = FPLLL_MINOR_VERSION_c
micro_version         = FPLLL_MICRO_VERSION_c
version               = "{0}.{1}.{2}".format(major_version, minor_version, micro_version)

max_enum_dim          = FPLLL_MAX_ENUM_DIM_c
have_recursive_enum   = FPLLL_HAVE_RECURSIVE_ENUM_c
max_parallel_enum_dim = FPLLL_MAX_PARALLEL_ENUM_DIM_c
