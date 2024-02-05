# -*- coding: utf-8 -*-
"""
Block Korkine Zolotarev algorithm.

..  moduleauthor:: Martin R.  Albrecht <martinralbrecht+fpylll@googlemail.com>
"""

include "fpylll/config.pxi"

from cysignals.signals cimport sig_on, sig_off

IF HAVE_QD:
    from .decl cimport mat_gso_mpz_dd, mat_gso_mpz_qd
    from .decl cimport mat_gso_long_dd, mat_gso_long_qd
    from .decl cimport dd_t, qd_t

from .decl cimport d_t

IF HAVE_LONG_DOUBLE:
    from  .decl cimport ld_t

from .decl cimport mat_gso_mpz_d, mat_gso_mpz_ld, mat_gso_mpz_dpe, mat_gso_mpz_mpfr, vector_fp_nr_t, fp_nr_t
from .decl cimport mat_gso_long_d, mat_gso_long_ld, mat_gso_long_dpe, mat_gso_long_mpfr
from .decl cimport mat_gso_gso_t, mat_gso_gram_t
from .fplll cimport BKZAutoAbort as BKZAutoAbort_c
from .fplll cimport BKZReduction as BKZReduction_c
from .fplll cimport BKZ_MAX_LOOPS, BKZ_MAX_TIME, BKZ_DUMP_GSO, BKZ_DEFAULT
from .fplll cimport BKZ_VERBOSE, BKZ_NO_LLL, BKZ_BOUNDED_LLL, BKZ_GH_BND, BKZ_AUTO_ABORT
from .fplll cimport BKZ_DEF_AUTO_ABORT_SCALE, BKZ_DEF_AUTO_ABORT_MAX_NO_DEC
from .fplll cimport BKZ_DEF_GH_FACTOR, BKZ_DEF_MIN_SUCCESS_PROBABILITY
from .fplll cimport BKZ_DEF_RERANDOMIZATION_DENSITY
from .fplll cimport BKZ_SD_VARIANT, BKZ_SLD_RED
from .fplll cimport MatGSOInterface as MatGSOInterface_c

from .fplll cimport FP_NR, Z_NR
from .fplll cimport FloatType
from .fplll cimport RED_BKZ_LOOPS_LIMIT, RED_BKZ_TIME_LIMIT
from .fplll cimport bkz_reduction as bkz_reduction_c
from .fplll cimport dpe_t
from .fplll cimport get_red_status_str
from .fplll cimport ZT_MPZ
from fpylll.gmp.mpz cimport mpz_t
from fpylll.mpfr.mpfr cimport mpfr_t
from fpylll.util cimport check_delta, check_precision, check_float_type
from fpylll.util import ReductionError
from .integer_matrix cimport IntegerMatrix
from fpylll.config import default_strategy, default_strategy_path

from .bkz_param import BKZEasyParam

cdef class BKZAutoAbort:
    """
    Utility class for aborting BKZ when slope does not improve any longer.
    """
    def __init__(self, MatGSO M, int num_rows, int start_row=0):
        """
        Create new auto abort object.

        :param M: GSO matrix
        :param num_rows: number of rows
        :param start_row: start at this row

        """

        if M._type == mat_gso_mpz_d:
            self._type = mat_gso_mpz_d
            self._core.mpz_d = new BKZAutoAbort_c[Z_NR[mpz_t], FP_NR[d_t]]((<MatGSOInterface_c[Z_NR[mpz_t], FP_NR[d_t]]*>M._core.mpz_d)[0],
                                                                           num_rows,
                                                                           start_row)
        elif M._type == mat_gso_long_d:
            self._type = mat_gso_long_d
            self._core.long_d = new BKZAutoAbort_c[Z_NR[long], FP_NR[d_t]]((<MatGSOInterface_c[Z_NR[long], FP_NR[d_t]]*>M._core.long_d)[0],
                                                                           num_rows,
                                                                           start_row)
        elif M._type == mat_gso_mpz_ld:
            IF HAVE_LONG_DOUBLE:
                self._type = mat_gso_mpz_ld
                self._core.mpz_ld = new BKZAutoAbort_c[Z_NR[mpz_t], FP_NR[ld_t]]((<MatGSOInterface_c[Z_NR[mpz_t], FP_NR[ld_t]]*>M._core.mpz_ld)[0],
                                                                                 num_rows,
                                                                                 start_row)
        elif M._type == mat_gso_long_ld:
            IF HAVE_LONG_DOUBLE:
                self._type = mat_gso_long_ld
                self._core.long_ld = new BKZAutoAbort_c[Z_NR[long], FP_NR[longdouble]]((<MatGSOInterface_c[Z_NR[long], FP_NR[ld_t]]*>M._core.long_ld)[0],
                                                                                       num_rows,
                                                                                       start_row)
            ELSE:
                raise RuntimeError("BKZAutoAbort object '%s' has no core."%self)
        elif M._type == mat_gso_mpz_dpe:
            self._type = mat_gso_mpz_dpe
            self._core.mpz_dpe = new BKZAutoAbort_c[Z_NR[mpz_t], FP_NR[dpe_t]]((<MatGSOInterface_c[Z_NR[mpz_t], FP_NR[dpe_t]]*>M._core.mpz_dpe)[0],
                                                                               num_rows,
                                                                               start_row)
        elif M._type == mat_gso_long_dpe:
            self._type = mat_gso_long_dpe
            self._core.long_dpe = new BKZAutoAbort_c[Z_NR[long], FP_NR[dpe_t]]((<MatGSOInterface_c[Z_NR[long], FP_NR[dpe_t]]*>M._core.long_dpe)[0],
                                                                               num_rows,
                                                                               start_row)
        elif M._type == mat_gso_mpz_mpfr:
            self._type = mat_gso_mpz_mpfr
            self._core.mpz_mpfr = new BKZAutoAbort_c[Z_NR[mpz_t], FP_NR[mpfr_t]]((<MatGSOInterface_c[Z_NR[mpz_t], FP_NR[mpfr_t]]*>M._core.mpz_mpfr)[0],
                                                                                 num_rows,
                                                                                 start_row)
        elif M._type == mat_gso_long_mpfr:
            self._type = mat_gso_long_mpfr
            self._core.long_mpfr = new BKZAutoAbort_c[Z_NR[long], FP_NR[mpfr_t]]((<MatGSOInterface_c[Z_NR[long], FP_NR[mpfr_t]]*>M._core.long_mpfr)[0],
                                                                                 num_rows,
                                                                                 start_row)
        else:
            IF HAVE_QD:
                if M._type == mat_gso_mpz_dd:
                    self._type = mat_gso_mpz_dd
                    self._core.mpz_dd = new BKZAutoAbort_c[Z_NR[mpz_t], FP_NR[dd_t]]((<MatGSOInterface_c[Z_NR[mpz_t], FP_NR[dd_t]]*>M._core.mpz_dd)[0],
                                                                                     num_rows,
                                                                                     start_row)
                elif M._type == mat_gso_long_dd:
                    self._type = mat_gso_long_dd
                    self._core.long_dd = new BKZAutoAbort_c[Z_NR[long], FP_NR[dd_t]]((<MatGSOInterface_c[Z_NR[long], FP_NR[dd_t]]*>M._core.long_dd)[0],
                                                                                     num_rows,
                                                                                     start_row)
                elif M._type == mat_gso_mpz_qd:
                    self._type = mat_gso_mpz_qd
                    self._core.mpz_qd = new BKZAutoAbort_c[Z_NR[mpz_t], FP_NR[qd_t]]((<MatGSOInterface_c[Z_NR[mpz_t], FP_NR[qd_t]]*>M._core.mpz_qd)[0],
                                                                                     num_rows,
                                                                                     start_row)
                elif M._type == mat_gso_long_qd:
                    self._type = mat_gso_long_qd
                    self._core.long_qd = new BKZAutoAbort_c[Z_NR[long], FP_NR[qd_t]]((<MatGSOInterface_c[Z_NR[long], FP_NR[qd_t]]*>M._core.long_qd)[0],
                                                                                     num_rows,
                                                                                     start_row)
                else:
                    raise RuntimeError("BKZAutoAbort object '%s' has no core."%self)
            ELSE:
                raise RuntimeError("BKZAutoAbort object '%s' has no core."%self)

        self.M = M

    def test_abort(self, scale=1.0, int max_no_dec=5):
        """
        Test if new slope fails to be smaller than `scale * old_slope`
        for `max_no_dec` iterations.

        :param scale: target decrease
        :param int max_no_dec: number of rounds allowed to be stuck
        """
        if self._type == mat_gso_mpz_d:
            return self._core.mpz_d.test_abort(scale, max_no_dec)
        elif self._type == mat_gso_long_d:
            return self._core.long_d.test_abort(scale, max_no_dec)
        elif self._type == mat_gso_mpz_ld:
            IF HAVE_LONG_DOUBLE:
                return self._core.mpz_ld.test_abort(scale, max_no_dec)
            ELSE:
                raise RuntimeError("BKZAutoAbort object '%s' has no core."%self)
        elif self._type == mat_gso_long_ld:
            IF HAVE_LONG_DOUBLE:
                return self._core.long_ld.test_abort(scale, max_no_dec)
            ELSE:
                raise RuntimeError("BKZAutoAbort object '%s' has no core."%self)
        elif self._type == mat_gso_mpz_dpe:
            return self._core.mpz_dpe.test_abort(scale, max_no_dec)
        elif self._type == mat_gso_long_dpe:
            return self._core.long_dpe.test_abort(scale, max_no_dec)
        elif self._type == mat_gso_mpz_mpfr:
            return self._core.mpz_mpfr.test_abort(scale, max_no_dec)
        elif self._type == mat_gso_long_mpfr:
            return self._core.long_mpfr.test_abort(scale, max_no_dec)
        else:
            IF HAVE_QD:
                if self._type == mat_gso_mpz_dd:
                    return self._core.mpz_dd.test_abort(scale, max_no_dec)
                elif self._type == mat_gso_long_dd:
                    return self._core.long_dd.test_abort(scale, max_no_dec)
                elif self._type == mat_gso_mpz_qd:
                    return self._core.mpz_qd.test_abort(scale, max_no_dec)
                elif self._type == mat_gso_long_qd:
                    return self._core.long_qd.test_abort(scale, max_no_dec)

        raise RuntimeError("BKZAutoAbort object '%s' has no core."%self)


cdef class BKZReduction:
    def __init__(self, MatGSO M, LLLReduction lll_obj, BKZParam param):
        """Construct new BKZ object.

        :param M: GSO object
        :param lll_obj: LLL object called as a subroutine
        :param param: parameters

        """
        self.M = M
        self.lll_obj = lll_obj
        self.param = param
        self._type = M._type

        if M._type == mat_gso_mpz_d:
            self._type = mat_gso_mpz_d
            self._core.mpz_d = new BKZReduction_c[Z_NR[mpz_t], FP_NR[double]]((<MatGSOInterface_c[Z_NR[mpz_t], FP_NR[double]]*>self.M._core.mpz_d)[0],
                                                                              self.lll_obj._core.mpz_d[0],
                                                                              param.o[0])
        elif M._type == mat_gso_mpz_ld:
            IF HAVE_LONG_DOUBLE:
                self._type = mat_gso_mpz_ld
                self._core.mpz_ld = new BKZReduction_c[Z_NR[mpz_t], FP_NR[ld_t]]((<MatGSOInterface_c[Z_NR[mpz_t], FP_NR[ld_t]]*>self.M._core.mpz_ld)[0],
                                                                                 self.lll_obj._core.mpz_ld[0],
                                                                                 param.o[0])
            ELSE:
                raise RuntimeError("BKZAutoAbort object '%s' has no core."%self)
        elif M._type == mat_gso_mpz_dpe:
            self._type = mat_gso_mpz_dpe
            self._core.mpz_dpe = new BKZReduction_c[Z_NR[mpz_t], FP_NR[dpe_t]]((<MatGSOInterface_c[Z_NR[mpz_t], FP_NR[dpe_t]]*>self.M._core.mpz_dpe)[0],
                                                                               self.lll_obj._core.mpz_dpe[0],
                                                                               param.o[0])
        elif M._type == mat_gso_mpz_mpfr:
            self._type = mat_gso_mpz_mpfr
            self._core.mpz_mpfr = new BKZReduction_c[Z_NR[mpz_t], FP_NR[mpfr_t]]((<MatGSOInterface_c[Z_NR[mpz_t], FP_NR[mpfr_t]]*>self.M._core.mpz_mpfr)[0],
                                                                                 self.lll_obj._core.mpz_mpfr[0],
                                                                                 param.o[0])
        elif M._type == mat_gso_long_d:
            self._type = mat_gso_long_d
            self._core.long_d = new BKZReduction_c[Z_NR[long], FP_NR[double]]((<MatGSOInterface_c[Z_NR[long], FP_NR[double]]*>self.M._core.long_d)[0],
                                                                              self.lll_obj._core.long_d[0],
                                                                              param.o[0])
        elif M._type == mat_gso_long_ld:
            IF HAVE_LONG_DOUBLE:
                self._type = mat_gso_long_ld
                self._core.long_ld = new BKZReduction_c[Z_NR[long], FP_NR[ld_t]]((<MatGSOInterface_c[Z_NR[long], FP_NR[ld_t]]*>self.M._core.long_ld)[0],
                                                                                 self.lll_obj._core.long_ld[0],
                                                                                 param.o[0])
            ELSE:
                raise RuntimeError("BKZAutoAbort object '%s' has no core."%self)
        elif M._type == mat_gso_long_dpe:
            self._type = mat_gso_long_dpe
            self._core.long_dpe = new BKZReduction_c[Z_NR[long], FP_NR[dpe_t]]((<MatGSOInterface_c[Z_NR[long], FP_NR[dpe_t]]*>self.M._core.long_dpe)[0],
                                                                               self.lll_obj._core.long_dpe[0],
                                                                               param.o[0])
        elif M._type == mat_gso_long_mpfr:
            self._type = mat_gso_long_mpfr
            self._core.long_mpfr = new BKZReduction_c[Z_NR[long], FP_NR[mpfr_t]]((<MatGSOInterface_c[Z_NR[long], FP_NR[mpfr_t]]*>self.M._core.long_mpfr)[0],
                                                                                 self.lll_obj._core.long_mpfr[0],
                                                                                 param.o[0])
        else:
            IF HAVE_QD:
                if M._type == mat_gso_mpz_dd:
                    self._type = mat_gso_mpz_dd
                    self._core.mpz_dd = new BKZReduction_c[Z_NR[mpz_t], FP_NR[dd_t]]((<MatGSOInterface_c[Z_NR[mpz_t], FP_NR[dd_t]]*>self.M._core.mpz_dd)[0],
                                                                           self.lll_obj._core.mpz_dd[0],
                                                                           param.o[0])
                elif M._type == mat_gso_mpz_qd:
                    self._type = mat_gso_mpz_qd
                    self._core.mpz_qd = new BKZReduction_c[Z_NR[mpz_t], FP_NR[qd_t]]((<MatGSOInterface_c[Z_NR[mpz_t], FP_NR[qd_t]]*>self.M._core.mpz_qd)[0],
                                                                           self.lll_obj._core.mpz_qd[0],
                                                                           param.o[0])
                elif M._type == mat_gso_long_dd:
                    self._type = mat_gso_long_dd
                    self._core.long_dd = new BKZReduction_c[Z_NR[long], FP_NR[dd_t]]((<MatGSOInterface_c[Z_NR[long], FP_NR[dd_t]]*>self.M._core.long_dd)[0],
                                                                           self.lll_obj._core.long_dd[0],
                                                                           param.o[0])
                elif M._type == mat_gso_long_qd:
                    self._type = mat_gso_long_qd
                    self._core.long_qd = new BKZReduction_c[Z_NR[long], FP_NR[qd_t]]((<MatGSOInterface_c[Z_NR[long], FP_NR[qd_t]]*>self.M._core.long_qd)[0],
                                                                           self.lll_obj._core.long_qd[0],
                                                                           param.o[0])
                else:
                    raise RuntimeError("MatGSO object '%s' has no core."%M)
            ELSE:
                raise RuntimeError("MatGSO object '%s' has no core."%M)

    def __dealloc__(self):
        if self._type == mat_gso_mpz_d:
            del self._core.mpz_d
        IF HAVE_LONG_DOUBLE:
            if self._type == mat_gso_mpz_ld:
                del self._core.mpz_ld
        if self._type == mat_gso_mpz_dpe:
            del self._core.mpz_dpe
        IF HAVE_QD:
            if self._type == mat_gso_mpz_dd:
                del self._core.mpz_dd
            if self._type == mat_gso_mpz_qd:
                del self._core.mpz_qd
        if self._type == mat_gso_mpz_mpfr:
            del self._core.mpz_mpfr
        if self._type == mat_gso_long_d:
            del self._core.long_d
        IF HAVE_LONG_DOUBLE:
            if self._type == mat_gso_long_ld:
                del self._core.long_ld
        if self._type == mat_gso_long_dpe:
            del self._core.long_dpe
        IF HAVE_QD:
            if self._type == mat_gso_long_dd:
                del self._core.long_dd
            if self._type == mat_gso_long_qd:
                del self._core.long_qd
        if self._type == mat_gso_long_mpfr:
            del self._core.long_mpfr

    def __reduce__(self):
        """
        Make sure attempts at pickling raise an error until proper pickling is implemented.
        """
        raise NotImplementedError

    def __call__(self):
        """
        Call BKZ, SD-BKZ or slide reduction.

        ..  note ::

            To enable the latter, set flags ``BKZ.SLD_RED`` or ``BKZ.SD_VARIANT`` when calling
            the constructor of this class.

        """
        if self._type == mat_gso_mpz_d:
            sig_on()
            r = self._core.mpz_d.bkz()
            sig_off()
        elif self._type == mat_gso_mpz_ld:
            IF HAVE_LONG_DOUBLE:
                sig_on()
                r = self._core.mpz_ld.bkz()
                sig_off()
            ELSE:
                raise RuntimeError("BKZAutoAbort object '%s' has no core."%self)
        elif self._type == mat_gso_mpz_dpe:
            sig_on()
            r = self._core.mpz_dpe.bkz()
            sig_off()
        elif self._type == mat_gso_mpz_mpfr:
            sig_on()
            r= self._core.mpz_mpfr.bkz()
            sig_off()
        elif self._type == mat_gso_long_d:
            sig_on()
            r = self._core.long_d.bkz()
            sig_off()
        elif self._type == mat_gso_long_ld:
            IF HAVE_LONG_DOUBLE:
                sig_on()
                r = self._core.long_ld.bkz()
                sig_off()
            ELSE:
                raise RuntimeError("BKZAutoAbort object '%s' has no core."%self)
        elif self._type == mat_gso_long_dpe:
            sig_on()
            r = self._core.long_dpe.bkz()
            sig_off()
        elif self._type == mat_gso_long_mpfr:
            sig_on()
            r= self._core.long_mpfr.bkz()
            sig_off()
        else:
            IF HAVE_QD:
                if self._type == mat_gso_mpz_dd:
                    sig_on()
                    r = self._core.mpz_dd.bkz()
                    sig_off()
                elif self._type == mat_gso_mpz_qd:
                    sig_on()
                    r = self._core.mpz_qd.bkz()
                    sig_off()
                elif self._type == mat_gso_long_dd:
                    sig_on()
                    r = self._core.long_dd.bkz()
                    sig_off()
                elif self._type == mat_gso_long_qd:
                    sig_on()
                    r = self._core.long_qd.bkz()
                    sig_off()
                else:
                    raise RuntimeError("BKZReduction object '%s' has no core."%self)
        return bool(r)

    def svp_preprocessing(self, int kappa, int block_size, BKZParam param):
        """Preprocess before calling (Dual-)SVP oracle.

        :param kappa: index
        :param block_size: block size
        :param param: reduction parameters

        """
        if kappa < 0 or kappa >= self.M.d:
            raise ValueError("kappa %d out of bounds (0, %d)"%(kappa, self.M.d))
        if block_size < 2 or block_size > self.M.d:
            raise ValueError("block size %d out of bounds (2, %d)"%(block_size, self.M.d))

        r = True

        if self._type == mat_gso_mpz_d:
            sig_on()
            r = self._core.mpz_d.svp_preprocessing(kappa, block_size, param.o[0])
            sig_off()
        elif self._type == mat_gso_mpz_ld:
            IF HAVE_LONG_DOUBLE:
                sig_on()
                r = self._core.mpz_ld.svp_preprocessing(kappa, block_size, param.o[0])
                sig_off()
            ELSE:
                raise RuntimeError("BKZAutoAbort object '%s' has no core."%self)
        elif self._type == mat_gso_mpz_dpe:
            sig_on()
            r = self._core.mpz_dpe.svp_preprocessing(kappa, block_size, param.o[0])
            sig_off()
        elif self._type == mat_gso_mpz_mpfr:
            sig_on()
            r= self._core.mpz_mpfr.svp_preprocessing(kappa, block_size, param.o[0])
            sig_off()
        elif self._type == mat_gso_long_d:
            sig_on()
            r = self._core.long_d.svp_preprocessing(kappa, block_size, param.o[0])
            sig_off()
        elif self._type == mat_gso_long_ld:
            IF HAVE_LONG_DOUBLE:
                sig_on()
                r = self._core.long_ld.svp_preprocessing(kappa, block_size, param.o[0])
                sig_off()
            ELSE:
                raise RuntimeError("BKZAutoAbort object '%s' has no core."%self)
        elif self._type == mat_gso_long_dpe:
            sig_on()
            r = self._core.long_dpe.svp_preprocessing(kappa, block_size, param.o[0])
            sig_off()
        elif self._type == mat_gso_long_mpfr:
            sig_on()
            r= self._core.long_mpfr.svp_preprocessing(kappa, block_size, param.o[0])
            sig_off()
        else:
            IF HAVE_QD:
                if self._type == mat_gso_mpz_dd:
                    sig_on()
                    r = self._core.mpz_dd.svp_preprocessing(kappa, block_size, param.o[0])
                    sig_off()
                elif self._type == mat_gso_mpz_qd:
                    sig_on()
                    r = self._core.mpz_qd.svp_preprocessing(kappa, block_size, param.o[0])
                    sig_off()
                elif self._type == mat_gso_long_dd:
                    sig_on()
                    r = self._core.long_dd.svp_preprocessing(kappa, block_size, param.o[0])
                    sig_off()
                elif self._type == mat_gso_long_qd:
                    sig_on()
                    r = self._core.long_qd.svp_preprocessing(kappa, block_size, param.o[0])
                    sig_off()
                else:
                    raise RuntimeError("BKZReduction object '%s' has no core."%self)

        return bool(r)

    def svp_postprocessing(self, int kappa, int block_size, tuple solution):
        """Insert solution into basis after SVP oracle call

        :param kappa: index
        :param block_size: block size
        :param solution: solution to insert

        """
        cdef vector_fp_nr_t solution_
        cdef fp_nr_t t

        if kappa < 0 or kappa >= self.M.d:
            raise ValueError("kappa %d out of bounds (0, %d)"%(kappa, self.M.d))
        if block_size < 2 or block_size > self.M.d:
            raise ValueError("block size %d out of bounds (2, %d)"%(block_size, self.M.d))

        r = True

        if self._type == mat_gso_mpz_d:
            for s in solution:
                t.d = float(s)
                solution_.d.push_back(t.d)
            sig_on()
            r = self._core.mpz_d.svp_postprocessing(kappa, block_size, solution_.d)
            sig_off()
        elif self._type == mat_gso_mpz_ld:
            IF HAVE_LONG_DOUBLE:
                for s in solution:
                    t.ld = float(s)
                    solution_.ld.push_back(t.ld)
                sig_on()
                r = self._core.mpz_ld.svp_postprocessing(kappa, block_size, solution_.ld)
                sig_off()
            ELSE:
                raise RuntimeError("BKZReduction object '%s' has no core."%self)
        elif self._type == mat_gso_mpz_dpe:
            for s in solution:
                t.dpe = float(s)
                solution_.dpe.push_back(t.dpe)
            sig_on()
            r = self._core.mpz_dpe.svp_postprocessing(kappa, block_size, solution_.dpe)
            sig_off()
        elif self._type == mat_gso_mpz_mpfr:
            for s in solution:
                t.mpfr = float(s)
                solution_.mpfr.push_back(t.mpfr)
            sig_on()
            r= self._core.mpz_mpfr.svp_postprocessing(kappa, block_size, solution_.mpfr)
            sig_off()
        elif self._type == mat_gso_long_d:
            for s in solution:
                t.d = float(s)
                solution_.d.push_back(t.d)
            sig_on()
            r = self._core.long_d.svp_postprocessing(kappa, block_size, solution_.d)
            sig_off()
        elif self._type == mat_gso_long_ld:
            IF HAVE_LONG_DOUBLE:
                for s in solution:
                    t.ld = float(s)
                    solution_.ld.push_back(t.ld)
                sig_on()
                r = self._core.long_ld.svp_postprocessing(kappa, block_size, solution_.ld)
                sig_off()
            ELSE:
                raise RuntimeError("BKZReduction object '%s' has no core."%self)
        elif self._type == mat_gso_long_dpe:
            for s in solution:
                t.dpe = float(s)
                solution_.dpe.push_back(t.dpe)
            sig_on()
            r = self._core.long_dpe.svp_postprocessing(kappa, block_size, solution_.dpe)
            sig_off()
        elif self._type == mat_gso_long_mpfr:
            for s in solution:
                t.mpfr = float(s)
                solution_.mpfr.push_back(t.mpfr)
            sig_on()
            r= self._core.long_mpfr.svp_postprocessing(kappa, block_size, solution_.mpfr)
            sig_off()
        else:
            IF HAVE_QD:
                if self._type == mat_gso_mpz_dd:
                    for s in solution:
                        t.dd = float(s)
                        solution_.dd.push_back(t.dd)
                    sig_on()
                    r = self._core.mpz_dd.svp_postprocessing(kappa, block_size, solution_.dd)
                    sig_off()
                elif self._type == mat_gso_mpz_qd:
                    for s in solution:
                        t.qd = float(s)
                        solution_.qd.push_back(t.qd)
                    sig_on()
                    r = self._core.mpz_qd.svp_postprocessing(kappa, block_size, solution_.qd)
                    sig_off()
                elif self._type == mat_gso_long_dd:
                    for s in solution:
                        t.dd = float(s)
                        solution_.dd.push_back(t.dd)
                    sig_on()
                    r = self._core.long_dd.svp_postprocessing(kappa, block_size, solution_.dd)
                    sig_off()
                elif self._type == mat_gso_long_qd:
                    for s in solution:
                        t.qd = float(s)
                        solution_.qd.push_back(t.qd)
                    sig_on()
                    r = self._core.long_qd.svp_postprocessing(kappa, block_size, solution_.qd)
                    sig_off()
                else:
                    raise RuntimeError("BKZReduction object '%s' has no core."%self)

        return bool(r)

    def svp_reduction(self, int kappa, int block_size, BKZParam param, dual=False):
        """Run (Dual-)SVP reduction (incl. pre and postprocessing)

        :param kappa: index
        :param block_size: block size
        :param param: reduction parameters
        :param dual: dual or primal reduction

        """
        if kappa < 0 or kappa >= self.M.d:
            raise ValueError("kappa %d out of bounds (0, %d)"%(kappa, self.M.d))
        if block_size < 2 or block_size > self.M.d:
            raise ValueError("block size %d out of bounds (2, %d)"%(block_size, self.M.d))

        r = True

        if self._type == mat_gso_mpz_d:
            sig_on()
            r = self._core.mpz_d.svp_reduction(kappa, block_size, param.o[0], int(dual))
            sig_off()
        elif self._type == mat_gso_mpz_ld:
            IF HAVE_LONG_DOUBLE:
                sig_on()
                r = self._core.mpz_ld.svp_reduction(kappa, block_size, param.o[0], dual)
                sig_off()
            ELSE:
                raise RuntimeError("BKZReduction object '%s' has no core."%self)
        elif self._type == mat_gso_mpz_dpe:
            sig_on()
            r = self._core.mpz_dpe.svp_reduction(kappa, block_size, param.o[0], dual)
            sig_off()
        elif self._type == mat_gso_mpz_mpfr:
            sig_on()
            r= self._core.mpz_mpfr.svp_reduction(kappa, block_size, param.o[0], dual)
            sig_off()
        elif self._type == mat_gso_long_d:
            sig_on()
            r = self._core.long_d.svp_reduction(kappa, block_size, param.o[0], int(dual))
            sig_off()
        elif self._type == mat_gso_long_ld:
            IF HAVE_LONG_DOUBLE:
                sig_on()
                r = self._core.long_ld.svp_reduction(kappa, block_size, param.o[0], dual)
                sig_off()
            ELSE:
                raise RuntimeError("BKZReduction object '%s' has no core."%self)
        elif self._type == mat_gso_long_dpe:
            sig_on()
            r = self._core.long_dpe.svp_reduction(kappa, block_size, param.o[0], dual)
            sig_off()
        elif self._type == mat_gso_long_mpfr:
            sig_on()
            r= self._core.long_mpfr.svp_reduction(kappa, block_size, param.o[0], dual)
            sig_off()
        else:
            IF HAVE_QD:
                if self._type == mat_gso_mpz_dd:
                    sig_on()
                    r = self._core.mpz_dd.svp_reduction(kappa, block_size, param.o[0], dual)
                    sig_off()
                elif self._type == mat_gso_mpz_qd:
                    sig_on()
                    r = self._core.mpz_qd.svp_reduction(kappa, block_size, param.o[0], dual)
                    sig_off()
                elif self._type == mat_gso_long_dd:
                    sig_on()
                    r = self._core.long_dd.svp_reduction(kappa, block_size, param.o[0], dual)
                    sig_off()
                elif self._type == mat_gso_long_qd:
                    sig_on()
                    r = self._core.long_qd.svp_reduction(kappa, block_size, param.o[0], dual)
                    sig_off()
                else:
                    raise RuntimeError("BKZReduction object '%s' has no core."%self)
        return bool(r)

    def tour(self, int loop, BKZParam param, int min_row, int max_row):
        """One BKZ tour.

        :param loop: loop index
        :param param: reduction parameters
        :param min_row: start row
        :param max_row: maximum row to consider (exclusive)
        :returns: tuple ``(clean, max_kappa)`` where ``clean == True``
                  if no changes were made, and ``max_kappa`` is the
                  maximum index for which no changes were made.
        """
        if min_row < 0 or min_row >= self.M.d:
            raise ValueError("min row %d out of bounds (0, %d)"%(min_row, self.M.d))
        if max_row < min_row or max_row > self.M.d:
            raise ValueError("max row %d out of bounds (%d, %d)"%(max_row, min_row, self.M.d))

        r = True
        cdef int kappa_max = 0
        if self._type == mat_gso_mpz_d:
            sig_on()
            r = self._core.mpz_d.tour(loop, kappa_max, param.o[0], min_row, max_row)
            sig_off()
        elif self._type == mat_gso_mpz_ld:
            IF HAVE_LONG_DOUBLE:
                sig_on()
                r = self._core.mpz_ld.tour(loop, kappa_max, param.o[0], min_row, max_row)
                sig_off()
            ELSE:
                raise RuntimeError("BKZReduction object '%s' has no core."%self)
        elif self._type == mat_gso_mpz_dpe:
            sig_on()
            r = self._core.mpz_dpe.tour(loop, kappa_max, param.o[0], min_row, max_row)
            sig_off()
        elif self._type == mat_gso_mpz_mpfr:
            sig_on()
            r= self._core.mpz_mpfr.tour(loop, kappa_max, param.o[0], min_row, max_row)
            sig_off()
        elif self._type == mat_gso_long_d:
            sig_on()
            r = self._core.long_d.tour(loop, kappa_max, param.o[0], min_row, max_row)
            sig_off()
        elif self._type == mat_gso_long_ld:
            IF HAVE_LONG_DOUBLE:
                sig_on()
                r = self._core.long_ld.tour(loop, kappa_max, param.o[0], min_row, max_row)
                sig_off()
            ELSE:
                raise RuntimeError("BKZReduction object '%s' has no core."%self)
        elif self._type == mat_gso_long_dpe:
            sig_on()
            r = self._core.long_dpe.tour(loop, kappa_max, param.o[0], min_row, max_row)
            sig_off()
        elif self._type == mat_gso_long_mpfr:
            sig_on()
            r= self._core.long_mpfr.tour(loop, kappa_max, param.o[0], min_row, max_row)
            sig_off()
        else:
            IF HAVE_QD:
                if self._type == mat_gso_mpz_dd:
                    sig_on()
                    r = self._core.mpz_dd.tour(loop, kappa_max, param.o[0], min_row, max_row)
                    sig_off()
                elif self._type == mat_gso_mpz_qd:
                    sig_on()
                    r = self._core.mpz_qd.tour(loop, kappa_max, param.o[0], min_row, max_row)
                    sig_off()
                elif self._type == mat_gso_long_dd:
                    sig_on()
                    r = self._core.long_dd.tour(loop, kappa_max, param.o[0], min_row, max_row)
                    sig_off()
                elif self._type == mat_gso_long_qd:
                    sig_on()
                    r = self._core.long_qd.tour(loop, kappa_max, param.o[0], min_row, max_row)
                    sig_off()
                else:
                    raise RuntimeError("BKZReduction object '%s' has no core."%self)
        return bool(r), kappa_max

    def sd_tour(self, int loop, BKZParam param, int min_row, int max_row):
        """One Dual-BKZ tour.

        :param loop: loop index
        :param param: reduction parameters
        :param min_row: start row
        :param max_row: maximum row to consider (exclusive)

        :returns: ``True`` if no changes were made, ``False`` otherwise.
        """
        if min_row < 0 or min_row >= self.M.d:
            raise ValueError("min row %d out of bounds (0, %d)"%(min_row, self.M.d))
        if max_row < min_row or max_row > self.M.d:
            raise ValueError("max row %d out of bounds (%d, %d)"%(max_row, min_row, self.M.d))

        r = True

        if self._type == mat_gso_mpz_d:
            sig_on()
            r = self._core.mpz_d.sd_tour(loop, param.o[0], min_row, max_row)
            sig_off()
        elif self._type == mat_gso_mpz_ld:
            IF HAVE_LONG_DOUBLE:
                sig_on()
                r = self._core.mpz_ld.sd_tour(loop, param.o[0], min_row, max_row)
                sig_off()
            ELSE:
                raise RuntimeError("BKZReduction object '%s' has no core."%self)
        elif self._type == mat_gso_mpz_dpe:
            sig_on()
            r = self._core.mpz_dpe.sd_tour(loop, param.o[0], min_row, max_row)
            sig_off()
        elif self._type == mat_gso_mpz_mpfr:
            sig_on()
            r= self._core.mpz_mpfr.sd_tour(loop, param.o[0], min_row, max_row)
            sig_off()
        elif self._type == mat_gso_long_d:
            sig_on()
            r = self._core.long_d.sd_tour(loop, param.o[0], min_row, max_row)
            sig_off()
        elif self._type == mat_gso_long_ld:
            IF HAVE_LONG_DOUBLE:
                sig_on()
                r = self._core.long_ld.sd_tour(loop, param.o[0], min_row, max_row)
                sig_off()
            ELSE:
                raise RuntimeError("BKZReduction object '%s' has no core."%self)
        elif self._type == mat_gso_long_dpe:
            sig_on()
            r = self._core.long_dpe.sd_tour(loop, param.o[0], min_row, max_row)
            sig_off()
        elif self._type == mat_gso_long_mpfr:
            sig_on()
            r= self._core.long_mpfr.sd_tour(loop, param.o[0], min_row, max_row)
            sig_off()
        else:
            IF HAVE_QD:
                if self._type == mat_gso_mpz_dd:
                    sig_on()
                    r = self._core.mpz_dd.sd_tour(loop, param.o[0], min_row, max_row)
                    sig_off()
                elif self._type == mat_gso_mpz_qd:
                    sig_on()
                    r = self._core.mpz_qd.sd_tour(loop, param.o[0], min_row, max_row)
                    sig_off()
                elif self._type == mat_gso_long_dd:
                    sig_on()
                    r = self._core.long_dd.sd_tour(loop, param.o[0], min_row, max_row)
                    sig_off()
                elif self._type == mat_gso_long_qd:
                    sig_on()
                    r = self._core.long_qd.sd_tour(loop, param.o[0], min_row, max_row)
                    sig_off()
                else:
                    raise RuntimeError("BKZReduction object '%s' has no core."%self)

        return bool(r)

    def slide_tour(self, int loop, BKZParam param, int min_row, int max_row):
        """One slide reduction tour.

        :param loop: loop index
        :param param: reduction parameters
        :param min_row: start row
        :param max_row: maximum row to consider (exclusive)

        :returns: ``True`` if no changes were made, ``False`` otherwise.

        ..  note ::

            You must run ``lll_obj()`` before calling this function, otherwise
            this function will produce an error.
        """
        if min_row < 0 or min_row >= self.M.d:
            raise ValueError("min row %d out of bounds (0, %d)"%(min_row, self.M.d))
        if max_row < min_row or max_row > self.M.d:
            raise ValueError("max row %d out of bounds (%d, %d)"%(max_row, min_row, self.M.d))

        r = True

        if self._type == mat_gso_mpz_d:
            sig_on()
            r = self._core.mpz_d.slide_tour(loop, param.o[0], min_row, max_row)
            sig_off()
        elif self._type == mat_gso_mpz_ld:
            IF HAVE_LONG_DOUBLE:
                sig_on()
                r = self._core.mpz_ld.slide_tour(loop, param.o[0], min_row, max_row)
                sig_off()
            ELSE:
                raise RuntimeError("BKZReduction object '%s' has no core."%self)
        elif self._type == mat_gso_mpz_dpe:
            sig_on()
            r = self._core.mpz_dpe.slide_tour(loop, param.o[0], min_row, max_row)
            sig_off()
        elif self._type == mat_gso_mpz_mpfr:
            sig_on()
            r= self._core.mpz_mpfr.slide_tour(loop, param.o[0], min_row, max_row)
            sig_off()
        elif self._type == mat_gso_long_d:
            sig_on()
            r = self._core.long_d.slide_tour(loop, param.o[0], min_row, max_row)
            sig_off()
        elif self._type == mat_gso_long_ld:
            IF HAVE_LONG_DOUBLE:
                sig_on()
                r = self._core.long_ld.slide_tour(loop, param.o[0], min_row, max_row)
                sig_off()
            ELSE:
                raise RuntimeError("BKZReduction object '%s' has no core."%self)
        elif self._type == mat_gso_long_dpe:
            sig_on()
            r = self._core.long_dpe.slide_tour(loop, param.o[0], min_row, max_row)
            sig_off()
        elif self._type == mat_gso_long_mpfr:
            sig_on()
            r= self._core.long_mpfr.slide_tour(loop, param.o[0], min_row, max_row)
            sig_off()
        else:
            IF HAVE_QD:
                if self._type == mat_gso_mpz_dd:
                    sig_on()
                    r = self._core.mpz_dd.slide_tour(loop, param.o[0], min_row, max_row)
                    sig_off()
                elif self._type == mat_gso_mpz_qd:
                    sig_on()
                    r = self._core.mpz_qd.slide_tour(loop, param.o[0], min_row, max_row)
                    sig_off()
                elif self._type == mat_gso_long_dd:
                    sig_on()
                    r = self._core.long_dd.slide_tour(loop, param.o[0], min_row, max_row)
                    sig_off()
                elif self._type == mat_gso_long_qd:
                    sig_on()
                    r = self._core.long_qd.slide_tour(loop, param.o[0], min_row, max_row)
                    sig_off()
                else:
                    raise RuntimeError("BKZReduction object '%s' has no core."%self)

        return bool(r)

    def hkz(self, BKZParam param, int min_row, int max_row):
        """HKZ reduction between ``min_row`` and ``max_row``.

        :param param: reduction parameters
        :param min_row: start row
        :param max_row: maximum row to consider (exclusive)

        :returns: ``True`` if no changes were made, ``False`` otherwise.

        """

        if min_row < 0 or min_row >= self.M.d:
            raise ValueError("min row %d out of bounds (0, %d)"%(min_row, self.M.d))
        if max_row < min_row or max_row > self.M.d:
            raise ValueError("max row %d out of bounds (%d, %d)"%(max_row, min_row, self.M.d))

        r = True
        cdef int kappa_max = 0

        if self._type == mat_gso_mpz_d:
            sig_on()
            r = self._core.mpz_d.hkz(kappa_max, param.o[0], min_row, max_row)
            sig_off()
        elif self._type == mat_gso_mpz_ld:
            IF HAVE_LONG_DOUBLE:
                sig_on()
                r = self._core.mpz_ld.hkz(kappa_max, param.o[0], min_row, max_row)
                sig_off()
            ELSE:
                raise RuntimeError("BKZReduction object '%s' has no core."%self)
        elif self._type == mat_gso_mpz_dpe:
            sig_on()
            r = self._core.mpz_dpe.hkz(kappa_max, param.o[0], min_row, max_row)
            sig_off()
        elif self._type == mat_gso_mpz_mpfr:
            sig_on()
            r= self._core.mpz_mpfr.hkz(kappa_max, param.o[0], min_row, max_row)
            sig_off()
        elif self._type == mat_gso_long_d:
            sig_on()
            r = self._core.long_d.hkz(kappa_max, param.o[0], min_row, max_row)
            sig_off()
        elif self._type == mat_gso_long_ld:
            IF HAVE_LONG_DOUBLE:
                sig_on()
                r = self._core.long_ld.hkz(kappa_max, param.o[0], min_row, max_row)
                sig_off()
            ELSE:
                raise RuntimeError("BKZReduction object '%s' has no core."%self)
        elif self._type == mat_gso_long_dpe:
            sig_on()
            r = self._core.long_dpe.hkz(kappa_max, param.o[0], min_row, max_row)
            sig_off()
        elif self._type == mat_gso_long_mpfr:
            sig_on()
            r= self._core.long_mpfr.hkz(kappa_max, param.o[0], min_row, max_row)
            sig_off()
        else:
            IF HAVE_QD:
                if self._type == mat_gso_mpz_dd:
                    sig_on()
                    r = self._core.mpz_dd.hkz(kappa_max, param.o[0], min_row, max_row)
                    sig_off()
                elif self._type == mat_gso_mpz_qd:
                    sig_on()
                    r = self._core.mpz_qd.hkz(kappa_max, param.o[0], min_row, max_row)
                    sig_off()
                elif self._type == mat_gso_long_dd:
                    sig_on()
                    r = self._core.long_dd.hkz(kappa_max, param.o[0], min_row, max_row)
                    sig_off()
                elif self._type == mat_gso_long_qd:
                    sig_on()
                    r = self._core.long_qd.hkz(kappa_max, param.o[0], min_row, max_row)
                    sig_off()
                else:
                    raise RuntimeError("BKZReduction object '%s' has no core."%self)
        return bool(r), kappa_max

    def rerandomize_block(self, int min_row, int max_row, int density):
        """Rerandomize block between ``min_row`` and ``max_row`` with a transform of ``density``

        :param min_row:
        :param max_row:
        :param density:

        """
        if self._type == mat_gso_mpz_d:
            sig_on()
            self._core.mpz_d.rerandomize_block(min_row, max_row, density)
            sig_off()
        elif self._type == mat_gso_mpz_ld:
            IF HAVE_LONG_DOUBLE:
                sig_on()
                self._core.mpz_ld.rerandomize_block(min_row, max_row, density)
                sig_off()
            ELSE:
                raise RuntimeError("BKZReduction object '%s' has no core."%self)
        elif self._type == mat_gso_mpz_dpe:
            sig_on()
            self._core.mpz_dpe.rerandomize_block(min_row, max_row, density)
            sig_off()
        elif self._type == mat_gso_mpz_mpfr:
            sig_on()
            self._core.mpz_mpfr.rerandomize_block(min_row, max_row, density)
            sig_off()
        elif self._type == mat_gso_long_d:
            sig_on()
            self._core.long_d.rerandomize_block(min_row, max_row, density)
            sig_off()
        elif self._type == mat_gso_long_ld:
            IF HAVE_LONG_DOUBLE:
                sig_on()
                self._core.long_ld.rerandomize_block(min_row, max_row, density)
                sig_off()
            ELSE:
                raise RuntimeError("BKZReduction object '%s' has no core."%self)
        elif self._type == mat_gso_long_dpe:
            sig_on()
            self._core.long_dpe.rerandomize_block(min_row, max_row, density)
            sig_off()
        elif self._type == mat_gso_long_mpfr:
            sig_on()
            self._core.long_mpfr.rerandomize_block(min_row, max_row, density)
            sig_off()
        else:
            IF HAVE_QD:
                if self._type == mat_gso_mpz_dd:
                    sig_on()
                    self._core.mpz_dd.rerandomize_block(min_row, max_row, density)
                    sig_off()
                elif self._type == mat_gso_mpz_qd:
                    sig_on()
                    self._core.mpz_qd.rerandomize_block(min_row, max_row, density)
                    sig_off()
                elif self._type == mat_gso_long_dd:
                    sig_on()
                    self._core.long_dd.rerandomize_block(min_row, max_row, density)
                    sig_off()
                elif self._type == mat_gso_long_qd:
                    sig_on()
                    self._core.long_qd.rerandomize_block(min_row, max_row, density)
                    sig_off()
                else:
                    raise RuntimeError("BKZReduction object '%s' has no core."%self)

    @property
    def status(self):
        """
        Status of this reduction.
        """
        if self._type == mat_gso_mpz_d:
            return self._core.mpz_d.status
        elif self._type == mat_gso_mpz_ld:
            IF HAVE_LONG_DOUBLE:
                return self._core.mpz_ld.status
            ELSE:
                raise RuntimeError("BKZReduction object '%s' has no core."%self)
        elif self._type == mat_gso_mpz_dpe:
            return self._core.mpz_dpe.status
        elif self._type == mat_gso_mpz_mpfr:
            return self._core.mpz_mpfr.status
        elif self._type == mat_gso_long_d:
            return self._core.long_d.status
        elif self._type == mat_gso_long_ld:
            IF HAVE_LONG_DOUBLE:
                return self._core.long_ld.status
            ELSE:
                raise RuntimeError("BKZReduction object '%s' has no core."%self)
        elif self._type == mat_gso_long_dpe:
            return self._core.long_dpe.status
        elif self._type == mat_gso_long_mpfr:
            return self._core.long_mpfr.status
        else:
            IF HAVE_QD:
                if self._type == mat_gso_mpz_dd:
                    return self._core.mpz_dd.status
                elif self._type == mat_gso_mpz_qd:
                    return self._core.mpz_qd.status
                elif self._type == mat_gso_long_dd:
                    return self._core.long_dd.status
                elif self._type == mat_gso_long_qd:
                    return self._core.long_qd.status
                else:
                    raise RuntimeError("BKZReduction object '%s' has no core."%self)

    @property
    def nodes(self):
        """
        Total number of enumeration nodes visited during this reduction.
        """
        if self._type == mat_gso_mpz_d:
            return self._core.mpz_d.nodes
        elif self._type == mat_gso_mpz_ld:
            IF HAVE_LONG_DOUBLE:
                return self._core.mpz_ld.nodes
            ELSE:
                raise RuntimeError("BKZReduction object '%s' has no core."%self)
        elif self._type == mat_gso_mpz_dpe:
            return self._core.mpz_dpe.nodes
        elif self._type == mat_gso_mpz_mpfr:
            return self._core.mpz_mpfr.nodes
        elif self._type == mat_gso_long_d:
            return self._core.long_d.nodes
        elif self._type == mat_gso_long_ld:
            IF HAVE_LONG_DOUBLE:
                return self._core.long_ld.nodes
            ELSE:
                raise RuntimeError("BKZReduction object '%s' has no core."%self)
        elif self._type == mat_gso_long_dpe:
            return self._core.long_dpe.nodes
        elif self._type == mat_gso_long_mpfr:
            return self._core.long_mpfr.nodes
        else:
            IF HAVE_QD:
                if self._type == mat_gso_mpz_dd:
                    return self._core.mpz_dd.nodes
                elif self._type == mat_gso_mpz_qd:
                    return self._core.mpz_qd.nodes
                elif self._type == mat_gso_long_dd:
                    return self._core.long_dd.nodes
                elif self._type == mat_gso_long_qd:
                    return self._core.long_qd.nodes
                else:
                    raise RuntimeError("BKZReduction object '%s' has no core."%self)



def bkz_reduction(IntegerMatrix B, BKZParam o, U=None, float_type=None, int precision=0):
    """
    Run BKZ reduction.

    :param IntegerMatrix B: Integer matrix, modified in place.
    :param BKZParam o: BKZ parameters
    :param U: Transformation matrix or ``None``
    :param float_type: either ``None``: for automatic choice or an entry of `fpylll.config.float_types`
    :param precision: bit precision to use if ``float_type`` is ``'mpfr'``

    :returns: modified matrix ``B``

    Example::

        >>> from fpylll import FPLLL, IntegerMatrix, BKZ
        >>> FPLLL.set_random_seed(1337)
        >>> A = IntegerMatrix(9, 10)
        >>> A.randomize("intrel", bits=10)
        >>> print(BKZ.reduction(A, BKZ.Param(3)))
        [  0  0  0  0 -1 -1  0  1  1  0 ]
        [  0  0  0 -1  1 -1  0  0 -1  0 ]
        [  0 -1  1  1  0  0  0  0 -1  0 ]
        [  0 -1 -1  0 -1  0  0  1  0  1 ]
        [  1 -1  1 -1 -1  0 -1  0  1  1 ]
        [  2  1  0  1 -1  0  1  0  0  0 ]
        [  0 -1 -1  1  1  0 -1  1  1 -1 ]
        [ -2  0 -1  0  1 -1  1 -2  0  1 ]
        [ -1  1  1  0  0 -1 -2  0  0  2 ]

    We can obtain the transformation matrix as well::

        >>> import copy
        >>> from fpylll import FPLLL, IntegerMatrix, BKZ
        >>> FPLLL.set_random_seed(1337)
        >>> A = IntegerMatrix(9, 10)
        >>> A.randomize("intrel", bits=10)
        >>> B = copy.copy(A)
        >>> U = IntegerMatrix(9, 9)
        >>> print(BKZ.reduction(A, BKZ.Param(3), U=U))
        [  0  0  0  0 -1 -1  0  1  1  0 ]
        [  0  0  0 -1  1 -1  0  0 -1  0 ]
        [  0 -1  1  1  0  0  0  0 -1  0 ]
        [  0 -1 -1  0 -1  0  0  1  0  1 ]
        [  1 -1  1 -1 -1  0 -1  0  1  1 ]
        [  2  1  0  1 -1  0  1  0  0  0 ]
        [  0 -1 -1  1  1  0 -1  1  1 -1 ]
        [ -2  0 -1  0  1 -1  1 -2  0  1 ]
        [ -1  1  1  0  0 -1 -2  0  0  2 ]
        >>> A == U * B
        True

    """
    check_precision(precision)

    cdef FloatType float_type_ = check_float_type(float_type)
    cdef int r = 0

    if B._type != ZT_MPZ:
        raise NotImplementedError("C++ BKZ is not implemented over longs, try the Python version.")

    if U is not None and isinstance(U, IntegerMatrix):
        with nogil:
            sig_on()
            r = bkz_reduction_c((<IntegerMatrix>B)._core.mpz,
                                (<IntegerMatrix>U)._core.mpz,
                                o.o[0], float_type_, precision)
            sig_off()
    else:
        with nogil:
            sig_on()
            r = bkz_reduction_c((<IntegerMatrix>B)._core.mpz,
                                NULL, o.o[0], float_type_, precision)
            sig_off()


    if r and r not in (RED_BKZ_LOOPS_LIMIT, RED_BKZ_TIME_LIMIT):
        raise ReductionError( str(get_red_status_str(r)) )

    return B


class BKZ:
    DEFAULT = BKZ_DEFAULT
    VERBOSE = BKZ_VERBOSE
    NO_LLL = BKZ_NO_LLL
    BOUNDED_LLL = BKZ_BOUNDED_LLL
    GH_BND = BKZ_GH_BND
    AUTO_ABORT = BKZ_AUTO_ABORT
    MAX_LOOPS = BKZ_MAX_LOOPS
    MAX_TIME = BKZ_MAX_TIME
    DUMP_GSO = BKZ_DUMP_GSO
    SD_VARIANT = BKZ_SD_VARIANT
    SLD_RED = BKZ_SLD_RED

    Param = BKZParam
    EasyParam = staticmethod(BKZEasyParam)
    AutoAbort = BKZAutoAbort
    reduction = staticmethod(bkz_reduction)
    Reduction = BKZReduction

    DEFAULT_AUTO_ABORT_SCALE        = BKZ_DEF_AUTO_ABORT_SCALE
    DEFAULT_AUTO_ABORT_MAX_NO_DEC   = BKZ_DEF_AUTO_ABORT_MAX_NO_DEC
    DEFAULT_GH_FACTOR               = BKZ_DEF_GH_FACTOR
    DEFAULT_MIN_SUCCESS_PROBABILITY = BKZ_DEF_MIN_SUCCESS_PROBABILITY
    DEFAULT_RERANDOMIZATION_DENSITY = BKZ_DEF_RERANDOMIZATION_DENSITY
    DEFAULT_STRATEGY = default_strategy
    DEFAULT_STRATEGY_PATH = default_strategy_path
