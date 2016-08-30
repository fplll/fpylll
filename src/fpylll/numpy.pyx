# -*- coding: utf-8 -*-
include "fpylll/config.pxi"
include "cysignals/signals.pxi"


from fpylll.fplll.gso cimport MatGSO
from fpylll.fplll.decl cimport mpz_double, mpz_ld, mpz_dpe, mpz_mpfr

IF HAVE_QD:
    from fpylll.fplll.decl cimport mpz_dd, mpz_qd

IF not HAVE_NUMPY:
    raise ImportError("NumPy is not installed, but this module relies on it.")

import numpy
from numpy.__init__ cimport ndarray  # TODO: that __init__ shouldn't be needed

def _dump_mu(ndarray[double, ndim=2, mode="c"] mu not None, MatGSO M, int kappa, int block_size):
    u"""
     Dump a block of the GSO matrix μ into a a numpy array.

     :param mu: numpy array of size (block_size*block_size) and type float64
     :param M: GSO object
     :param kappa: index of the beginning of the block
     :param block_size: size of the considered block

     :returns: Nothing
     """
    if M._type == mpz_double:
        return M._core.mpz_double.dump_mu_d(&mu[0,0], kappa, block_size)
    if M._type == mpz_ld:
        return M._core.mpz_ld.dump_mu_d(&mu[0,0], kappa, block_size)
    if M._type == mpz_dpe:
        return M._core.mpz_dpe.dump_mu_d(&mu[0,0], kappa, block_size)
    IF HAVE_QD:
        if M._type == mpz_dd:
            return M._core.mpz_dd.dump_mu_d(&mu[0,0], kappa, block_size)
        if M._type == mpz_qd:
            return M._core.mpz_qd.dump_mu_d(&mu[0,0], kappa, block_size)
    if M._type == mpz_mpfr:
        return M._core.mpz_mpfr.dump_mu_d(&mu[0,0], kappa, block_size)

    raise RuntimeError("MatGSO object '%s' has no core."%M)

def dump_mu(MatGSO m, int kappa, int block_size):
    u"""
     Dump a block of the GSO matrix μ into a a numpy array.

     :param M: GSO object
     :param kappa: index of the beginning of the block
     :param block_size: size of the considered block

     :returns: Nothing
     """
    mu = ndarray(dtype='float64', shape=(block_size, block_size))
    _dump_mu(mu, m, kappa, block_size)
    return mu

def _dump_r(ndarray[double, ndim=1, mode="c"] r not None, MatGSO M, int kappa, int block_size):
    u"""
     Dump a block of the GSO vector r into a a numpy array.

     :param mu: numpy array of size (block_size) and type float64
     :param M: GSO object
     :param kappa: index of the beginning of the block
     :param block_size: size of the considered block

     :returns: Nothing
     """

    if M._type == mpz_double:
        return M._core.mpz_double.dump_r_d(&r[0], kappa, block_size)
    if M._type == mpz_ld:
        return M._core.mpz_ld.dump_r_d(&r[0], kappa, block_size)
    if M._type == mpz_dpe:
        return M._core.mpz_dpe.dump_r_d(&r[0], kappa, block_size)
    IF HAVE_QD:
        if M._type == mpz_dd:
            return M._core.mpz_dd.dump_r_d(&r[0], kappa, block_size)
        if M._type == mpz_qd:
            return M._core.mpz_qd.dump_r_d(&r[0], kappa, block_size)
    if M._type == mpz_mpfr:
        return M._core.mpz_mpfr.dump_r_d(&r[0], kappa, block_size)

    raise RuntimeError("MatGSO object '%s' has no core."%M)

def dump_r(MatGSO M, int kappa, int block_size):
    u"""
     Dump a block of the GSO vector r into a a numpy array.

     :param M: GSO object
     :param kappa: index of the beginning of the block
     :param block_size: size of the considered block

     :returns: Nothing
     """
    r = ndarray(dtype='float64', shape=block_size)
    _dump_r(r, M, kappa, block_size)
    return r
