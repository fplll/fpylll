# -*- coding: utf-8 -*-
include "fpylll/config.pxi"


from fpylll.fplll.gso cimport MatGSO
from fpylll.fplll.decl cimport mat_gso_mpz_d, mat_gso_mpz_ld, mat_gso_mpz_dpe, mat_gso_mpz_mpfr
from fpylll.fplll.decl cimport mat_gso_long_d, mat_gso_long_ld, mat_gso_long_dpe, mat_gso_long_mpfr

IF HAVE_QD:
    from fpylll.fplll.decl cimport mat_gso_mpz_dd, mat_gso_mpz_qd
    from fpylll.fplll.decl cimport mat_gso_long_dd, mat_gso_long_qd

IF not HAVE_NUMPY:
    raise ImportError("NumPy is not installed, but this module relies on it.")

import numpy
from numpy.__init__ cimport ndarray  # TODO: that __init__ shouldn't be needed
from numpy.__init__ cimport integer as np_integer

def _dump_mu(ndarray[double, ndim=2, mode="c"] mu not None, MatGSO M, int kappa, int block_size):
    u"""
     Dump a block of the GSO matrix μ into a a numpy array.

     :param mu: numpy array of size (block_size*block_size) and type float64
     :param M: GSO object
     :param kappa: index of the beginning of the block
     :param block_size: size of the considered block

     :returns: Nothing
     """
    if M._type == mat_gso_mpz_d:
        return M._core.mpz_d.dump_mu_d(&mu[0,0], kappa, block_size)
    IF HAVE_LONG_DOUBLE:
        if M._type == mat_gso_mpz_ld:
            return M._core.mpz_ld.dump_mu_d(&mu[0,0], kappa, block_size)
    if M._type == mat_gso_mpz_dpe:
        return M._core.mpz_dpe.dump_mu_d(&mu[0,0], kappa, block_size)
    IF HAVE_QD:
        if M._type == mat_gso_mpz_dd:
            return M._core.mpz_dd.dump_mu_d(&mu[0,0], kappa, block_size)
        if M._type == mat_gso_mpz_qd:
            return M._core.mpz_qd.dump_mu_d(&mu[0,0], kappa, block_size)
    if M._type == mat_gso_mpz_mpfr:
        return M._core.mpz_mpfr.dump_mu_d(&mu[0,0], kappa, block_size)

    if M._type == mat_gso_long_d:
        return M._core.long_d.dump_mu_d(&mu[0,0], kappa, block_size)
    IF HAVE_LONG_DOUBLE:
        if M._type == mat_gso_long_ld:
            return M._core.long_ld.dump_mu_d(&mu[0,0], kappa, block_size)
    if M._type == mat_gso_long_dpe:
        return M._core.long_dpe.dump_mu_d(&mu[0,0], kappa, block_size)
    IF HAVE_QD:
        if M._type == mat_gso_long_dd:
            return M._core.long_dd.dump_mu_d(&mu[0,0], kappa, block_size)
        if M._type == mat_gso_long_qd:
            return M._core.long_qd.dump_mu_d(&mu[0,0], kappa, block_size)
    if M._type == mat_gso_long_mpfr:
        return M._core.long_mpfr.dump_mu_d(&mu[0,0], kappa, block_size)

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

    if M._type == mat_gso_mpz_d:
        return M._core.mpz_d.dump_r_d(&r[0], kappa, block_size)
    IF HAVE_LONG_DOUBLE:
        if M._type == mat_gso_mpz_ld:
            return M._core.mpz_ld.dump_r_d(&r[0], kappa, block_size)
    if M._type == mat_gso_mpz_dpe:
        return M._core.mpz_dpe.dump_r_d(&r[0], kappa, block_size)
    IF HAVE_QD:
        if M._type == mat_gso_mpz_dd:
            return M._core.mpz_dd.dump_r_d(&r[0], kappa, block_size)
        if M._type == mat_gso_mpz_qd:
            return M._core.mpz_qd.dump_r_d(&r[0], kappa, block_size)
    if M._type == mat_gso_mpz_mpfr:
        return M._core.mpz_mpfr.dump_r_d(&r[0], kappa, block_size)

    if M._type == mat_gso_long_d:
        return M._core.long_d.dump_r_d(&r[0], kappa, block_size)
    IF HAVE_LONG_DOUBLE:
        if M._type == mat_gso_long_ld:
            return M._core.long_ld.dump_r_d(&r[0], kappa, block_size)
    if M._type == mat_gso_long_dpe:
        return M._core.long_dpe.dump_r_d(&r[0], kappa, block_size)
    IF HAVE_QD:
        if M._type == mat_gso_long_dd:
            return M._core.long_dd.dump_r_d(&r[0], kappa, block_size)
        if M._type == mat_gso_long_qd:
            return M._core.long_qd.dump_r_d(&r[0], kappa, block_size)
    if M._type == mat_gso_long_mpfr:
        return M._core.long_mpfr.dump_r_d(&r[0], kappa, block_size)

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

def is_numpy_integer(value):
    """
    Return true if value is a numpy integer, false otherwise.
    :param value: the value to be checked.
    :returns: True if value is a numpy integer, false otherwise.
    """
    return isinstance(value, np_integer)
