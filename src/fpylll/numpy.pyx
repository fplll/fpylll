# -*- coding: utf-8 -*-
include "fpylll/config.pxi"
include "cysignals/signals.pxi"


from fpylll.fplll.gso cimport MatGSO
from fpylll.fplll.decl cimport mpz_double, mpz_ld, mpz_dpe, mpz_mpfr

IF HAVE_QD:
    from fpylll cimport mpz_dd, mpz_qd

IF not HAVE_NUMPY:
    raise ImportError("NumPy is not installed, but this module relies on it.")

import numpy
from numpy.__init__ cimport ndarray  # TODO: that __init__ shouldn't be needed

def dump_mu(ndarray[double, ndim=2, mode="c"] mu not None, MatGSO m, int kappa, int blocksize):
    u"""
     Dump a block of the GSO matrix μ into a a numpy array.

     :param mu: A numpy array of size (blocksize*blocksize) and type float64
     :param kappa: index of the beginning of the block
     :param blocksize: size of the considered block

     :returns: Nothing
     """
    if m._type == mpz_double:
        return m._core.mpz_double.dumpMu_d(&mu[0,0], kappa, blocksize)
    if m._type == mpz_ld:
        return m._core.mpz_ld.dumpMu_d(&mu[0,0], kappa, blocksize)
    if m._type == mpz_dpe:
        return m._core.mpz_dpe.dumpMu_d(&mu[0,0], kappa, blocksize)
    IF HAVE_QD:
        if m._type == mpz_dd:
            return m._core.mpz_dd.dumpMu_d(&mu[0,0], kappa, blocksize)
        if m._type == mpz_qd:
            return m._core.mpz_qd.dumpMu_d(&mu[0,0], kappa, blocksize)
    if m._type == mpz_mpfr:
        return m._core.mpz_mpfr.dumpMu_d(&mu[0,0], kappa, blocksize)

    raise RuntimeError("MatGSO object '%s' has no core."%m)


def dump_r(ndarray[double, ndim=1, mode="c"] r not None, MatGSO m, int kappa, int blocksize):
    u"""
     Dump a block of the GSO vector r into a a numpy array.

     :param mu: A numpy array of size (blocksize) and type float64
     :param kappa: index of the beginning of the block
     :param blocksize: size of the considered block

     :returns: Nothing
     """

    if m._type == mpz_double:
        return m._core.mpz_double.dumpR_d(&r[0], kappa, blocksize)
    if m._type == mpz_ld:
        return m._core.mpz_ld.dumpR_d(&r[0], kappa, blocksize)
    if m._type == mpz_dpe:
        return m._core.mpz_dpe.dumpR_d(&r[0], kappa, blocksize)
    IF HAVE_QD:
        if m._type == mpz_dd:
            return m._core.mpz_dd.dumpR_d(&r[0], kappa, blocksize)
        if m._type == mpz_qd:
            return m._core.mpz_qd.dumpR_d(&r[0], kappa, blocksize)
    if m._type == mpz_mpfr:
        return m._core.mpz_mpfr.dumpR_d(&r[0], kappa, blocksize)

    raise RuntimeError("MatGSO object '%s' has no core."%m)
