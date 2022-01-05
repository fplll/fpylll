# -*- coding: utf-8 -*-
include "fpylll/config.pxi"


from cysignals.signals cimport sig_on, sig_off

from fpylll.gmp.mpz cimport mpz_t
from fpylll.mpfr.mpfr cimport mpfr_t
from .integer_matrix cimport IntegerMatrix

from .fplll cimport LLL_VERBOSE
from .fplll cimport LLL_EARLY_RED
from .fplll cimport LLL_SIEGEL
from .fplll cimport LLL_DEFAULT

from .fplll cimport LLLMethod, LLL_DEF_ETA, LLL_DEF_DELTA
from .fplll cimport LM_WRAPPER, LM_PROVED, LM_HEURISTIC, LM_FAST
from .fplll cimport FT_DEFAULT, FT_DOUBLE, FT_LONG_DOUBLE, FT_DD, FT_QD
from .fplll cimport ZT_MPZ

from .fplll cimport dpe_t
from .fplll cimport Z_NR, FP_NR
from .fplll cimport lll_reduction as lll_reduction_c
from .fplll cimport RED_SUCCESS
from .fplll cimport MatGSOInterface as MatGSOInterface_c
from .fplll cimport LLLReduction as LLLReduction_c
from .fplll cimport get_red_status_str
from .fplll cimport is_lll_reduced
from .fplll cimport FloatType

from fpylll.util cimport check_float_type, check_delta, check_eta, check_precision
from fpylll.util import ReductionError

from .decl cimport d_t
from .decl cimport mat_gso_mpz_d, mat_gso_mpz_ld, mat_gso_mpz_dpe, mat_gso_mpz_mpfr
from .decl cimport mat_gso_long_d, mat_gso_long_ld, mat_gso_long_dpe, mat_gso_long_mpfr

IF HAVE_LONG_DOUBLE:
    from .decl cimport ld_t

IF HAVE_QD:
    from .decl cimport mat_gso_mpz_dd, mat_gso_mpz_qd, mat_gso_long_dd, mat_gso_long_qd, dd_t, qd_t

from .wrapper import Wrapper

cdef class LLLReduction:
    def __init__(self, MatGSO M,
                 double delta=LLL_DEF_DELTA,
                 double eta=LLL_DEF_ETA,
                 int flags=LLL_DEFAULT):
        """Construct new LLL object.

        :param MatGSO M:
        :param double delta:
        :param double eta:
        :param int flags:

            - ``DEFAULT``:

            - ``VERBOSE``:

            - ``EARLY_RED``:

            - ``SIEGEL``:

        """

        check_delta(delta)
        check_eta(eta)

        cdef MatGSOInterface_c[Z_NR[mpz_t], FP_NR[d_t]]  *m_mpz_double
        IF HAVE_LONG_DOUBLE:
            cdef MatGSOInterface_c[Z_NR[mpz_t], FP_NR[ld_t]] *m_mpz_ld
        cdef MatGSOInterface_c[Z_NR[mpz_t], FP_NR[dpe_t]] *m_mpz_dpe
        IF HAVE_QD:
            cdef MatGSOInterface_c[Z_NR[mpz_t], FP_NR[dd_t]] *m_mpz_dd
            cdef MatGSOInterface_c[Z_NR[mpz_t], FP_NR[qd_t]] *m_mpz_qd
        cdef MatGSOInterface_c[Z_NR[mpz_t], FP_NR[mpfr_t]]  *m_mpz_mpfr

        cdef MatGSOInterface_c[Z_NR[long], FP_NR[d_t]]  *m_long_double
        IF HAVE_LONG_DOUBLE:
            cdef MatGSOInterface_c[Z_NR[long], FP_NR[ld_t]] *m_long_ld
        cdef MatGSOInterface_c[Z_NR[long], FP_NR[dpe_t]] *m_long_dpe
        IF HAVE_QD:
            cdef MatGSOInterface_c[Z_NR[long], FP_NR[dd_t]] *m_long_dd
            cdef MatGSOInterface_c[Z_NR[long], FP_NR[qd_t]] *m_long_qd
        cdef MatGSOInterface_c[Z_NR[long], FP_NR[mpfr_t]]  *m_long_mpfr

        self.M = M

        if M._type == mat_gso_mpz_d:
            m_mpz_double = M._core.mpz_d
            self._type = mat_gso_mpz_d
            self._core.mpz_d = new LLLReduction_c[Z_NR[mpz_t], FP_NR[d_t]](m_mpz_double[0],
                                                                           delta,
                                                                           eta, flags)
        elif M._type == mat_gso_mpz_ld:
            IF HAVE_LONG_DOUBLE:
                m_mpz_ld = M._core.mpz_ld
                self._type = mat_gso_mpz_ld
                self._core.mpz_ld = new LLLReduction_c[Z_NR[mpz_t], FP_NR[ld_t]](m_mpz_ld[0],
                                                                                 delta,
                                                                                 eta, flags)
            ELSE:
                raise RuntimeError("MatGSO object '%s' has no core."%self)
        elif M._type == mat_gso_mpz_dpe:
            m_mpz_dpe = M._core.mpz_dpe
            self._type = mat_gso_mpz_dpe
            self._core.mpz_dpe = new LLLReduction_c[Z_NR[mpz_t], FP_NR[dpe_t]](m_mpz_dpe[0],
                                                                               delta,
                                                                               eta, flags)
        elif M._type == mat_gso_mpz_mpfr:
            m_mpz_mpfr = M._core.mpz_mpfr
            self._type = mat_gso_mpz_mpfr
            self._core.mpz_mpfr = new LLLReduction_c[Z_NR[mpz_t], FP_NR[mpfr_t]](m_mpz_mpfr[0],
                                                                                 delta,
                                                                                 eta, flags)
        elif M._type == mat_gso_long_d:
            m_long_double = M._core.long_d
            self._type = mat_gso_long_d
            self._core.long_d = new LLLReduction_c[Z_NR[long], FP_NR[d_t]](m_long_double[0],
                                                                           delta,
                                                                           eta, flags)
        elif M._type == mat_gso_long_ld:
            IF HAVE_LONG_DOUBLE:
                m_long_ld = M._core.long_ld
                self._type = mat_gso_long_ld
                self._core.long_ld = new LLLReduction_c[Z_NR[long], FP_NR[ld_t]](m_long_ld[0],
                                                                                 delta,
                                                                                 eta, flags)
            ELSE:
                raise RuntimeError("MatGSO object '%s' has no core."%self)
        elif M._type == mat_gso_long_dpe:
            m_long_dpe = M._core.long_dpe
            self._type = mat_gso_long_dpe
            self._core.long_dpe = new LLLReduction_c[Z_NR[long], FP_NR[dpe_t]](m_long_dpe[0],
                                                                               delta,
                                                                               eta, flags)
        elif M._type == mat_gso_long_mpfr:
            m_long_mpfr = M._core.long_mpfr
            self._type = mat_gso_long_mpfr
            self._core.long_mpfr = new LLLReduction_c[Z_NR[long], FP_NR[mpfr_t]](m_long_mpfr[0],
                                                                                 delta,
                                                                                 eta, flags)
        else:
            IF HAVE_QD:
                if M._type == mat_gso_mpz_dd:
                    m_mpz_dd = M._core.mpz_dd
                    self._type = mat_gso_mpz_dd
                    self._core.mpz_dd = new LLLReduction_c[Z_NR[mpz_t], FP_NR[dd_t]](m_mpz_dd[0],
                                                                                     delta,
                                                                                     eta, flags)
                elif M._type == mat_gso_mpz_qd:
                    m_mpz_qd = M._core.mpz_qd
                    self._type = mat_gso_mpz_qd
                    self._core.mpz_qd = new LLLReduction_c[Z_NR[mpz_t], FP_NR[qd_t]](m_mpz_qd[0],
                                                                                     delta,
                                                                                     eta, flags)
                elif M._type == mat_gso_long_dd:
                    m_long_dd = M._core.long_dd
                    self._type = mat_gso_long_dd
                    self._core.long_dd = new LLLReduction_c[Z_NR[long], FP_NR[dd_t]](m_long_dd[0],
                                                                                     delta,
                                                                                     eta, flags)
                elif M._type == mat_gso_long_qd:
                    m_long_qd = M._core.long_qd
                    self._type = mat_gso_long_qd
                    self._core.long_qd = new LLLReduction_c[Z_NR[long], FP_NR[qd_t]](m_long_qd[0],
                                                                                     delta,
                                                                                     eta, flags)
                else:
                    raise RuntimeError("MatGSO object '%s' has no core."%self)
            ELSE:
                raise RuntimeError("MatGSO object '%s' has no core."%self)

        self._delta = delta
        self._eta = eta

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

    def __call__(self, int kappa_min=0, int kappa_start=0, int kappa_end=-1, int size_reduction_start=0):
        """LLL reduction.

        :param int kappa_min: minimal index to go back to
        :param int kappa_start: index to start processing at
        :param int kappa_end: end index (exclusive)
        :param int size_reduction_start: only perform size reductions using vectors starting at this
            index
        """
        if self.M.d == 0:
            return

        if kappa_end == -1:
            kappa_end = self.M.d

        cdef int r
        if self._type == mat_gso_mpz_d:
            sig_on()
            self._core.mpz_d.lll(kappa_min, kappa_start, kappa_end, size_reduction_start)
            r = self._core.mpz_d.status
            sig_off()
        elif self._type == mat_gso_mpz_ld:
            IF HAVE_LONG_DOUBLE:
                sig_on()
                self._core.mpz_ld.lll(kappa_min, kappa_start, kappa_end, size_reduction_start)
                r = self._core.mpz_ld.status
                sig_off()
            ELSE:
                raise RuntimeError("LLLReduction object '%s' has no core."%self)
        elif self._type == mat_gso_mpz_dpe:
            sig_on()
            self._core.mpz_dpe.lll(kappa_min, kappa_start, kappa_end, size_reduction_start)
            r = self._core.mpz_dpe.status
            sig_off()
        elif self._type == mat_gso_mpz_mpfr:
            sig_on()
            self._core.mpz_mpfr.lll(kappa_min, kappa_start, kappa_end, size_reduction_start)
            r = self._core.mpz_mpfr.status
            sig_off()
        elif self._type == mat_gso_long_d:
            sig_on()
            self._core.long_d.lll(kappa_min, kappa_start, kappa_end, size_reduction_start)
            r = self._core.long_d.status
            sig_off()
        elif self._type == mat_gso_long_ld:
            IF HAVE_LONG_DOUBLE:
                sig_on()
                self._core.long_ld.lll(kappa_min, kappa_start, kappa_end, size_reduction_start)
                r = self._core.long_ld.status
                sig_off()
            ELSE:
                raise RuntimeError("LLLReduction object '%s' has no core."%self)
        elif self._type == mat_gso_long_dpe:
            sig_on()
            self._core.long_dpe.lll(kappa_min, kappa_start, kappa_end, size_reduction_start)
            r = self._core.long_dpe.status
            sig_off()
        elif self._type == mat_gso_long_mpfr:
            sig_on()
            self._core.long_mpfr.lll(kappa_min, kappa_start, kappa_end, size_reduction_start)
            r = self._core.long_mpfr.status
            sig_off()
        else:
            IF HAVE_QD:
                if self._type == mat_gso_mpz_dd:
                    sig_on()
                    self._core.mpz_dd.lll(kappa_min, kappa_start, kappa_end, size_reduction_start)
                    r = self._core.mpz_dd.status
                    sig_off()
                elif self._type == mat_gso_mpz_qd:
                    sig_on()
                    self._core.mpz_qd.lll(kappa_min, kappa_start, kappa_end, size_reduction_start)
                    r = self._core.mpz_qd.status
                    sig_off()
                elif self._type == mat_gso_long_dd:
                    sig_on()
                    self._core.long_dd.lll(kappa_min, kappa_start, kappa_end, size_reduction_start)
                    r = self._core.long_dd.status
                    sig_off()
                elif self._type == mat_gso_long_qd:
                    sig_on()
                    self._core.long_qd.lll(kappa_min, kappa_start, kappa_end, size_reduction_start)
                    r = self._core.long_qd.status
                    sig_off()
                else:
                    raise RuntimeError("LLLReduction object '%s' has no core."%self)
            ELSE:
                raise RuntimeError("LLLReduction object '%s' has no core."%self)

        if r:
            raise ReductionError( str(get_red_status_str(r)) )

    def size_reduction(self, int kappa_min=0, int kappa_end=-1, int size_reduction_start=0):
        """Size reduction.

        :param int kappa_min: start index
        :param int kappa_end: end index (exclusive)
        :param int size_reduction_start: only perform size reductions using vectors starting at this
            index
        """
        if kappa_end == -1:
            kappa_end = self.M.d

        if self._type == mat_gso_mpz_d:
            sig_on()
            r = self._core.mpz_d.size_reduction(kappa_min, kappa_end, size_reduction_start)
            sig_off()
        elif self._type == mat_gso_long_d:
            sig_on()
            r = self._core.long_d.size_reduction(kappa_min, kappa_end, size_reduction_start)
            sig_off()
        elif self._type == mat_gso_mpz_ld:
            IF HAVE_LONG_DOUBLE:
                sig_on()
                r = self._core.mpz_ld.size_reduction(kappa_min, kappa_end, size_reduction_start)
                sig_off()
            ELSE:
                raise RuntimeError("LLLReduction object '%s' has no core."%self)
        elif self._type == mat_gso_long_ld:
            IF HAVE_LONG_DOUBLE:
                sig_on()
                r = self._core.long_ld.size_reduction(kappa_min, kappa_end, size_reduction_start)
                sig_off()
            ELSE:
                raise RuntimeError("LLLReduction object '%s' has no core."%self)
        elif self._type == mat_gso_mpz_dpe:
            sig_on()
            r = self._core.mpz_dpe.size_reduction(kappa_min, kappa_end, size_reduction_start)
            sig_off()
        elif self._type == mat_gso_mpz_mpfr:
            sig_on()
            r = self._core.mpz_mpfr.size_reduction(kappa_min, kappa_end, size_reduction_start)
            sig_off()
        elif self._type == mat_gso_long_dpe:
            sig_on()
            r = self._core.long_dpe.size_reduction(kappa_min, kappa_end, size_reduction_start)
            sig_off()
        elif self._type == mat_gso_long_mpfr:
            sig_on()
            r = self._core.long_mpfr.size_reduction(kappa_min, kappa_end, size_reduction_start)
            sig_off()
        else:
            IF HAVE_QD:
                if self._type == mat_gso_mpz_dd:
                    sig_on()
                    r = self._core.mpz_dd.size_reduction(kappa_min, kappa_end, size_reduction_start)
                    sig_off()
                elif self._type == mat_gso_mpz_qd:
                    sig_on()
                    r = self._core.mpz_qd.size_reduction(kappa_min, kappa_end, size_reduction_start)
                    sig_off()
                elif self._type == mat_gso_long_dd:
                    sig_on()
                    r = self._core.long_dd.size_reduction(kappa_min, kappa_end, size_reduction_start)
                    sig_off()
                elif self._type == mat_gso_long_qd:
                    sig_on()
                    r = self._core.long_qd.size_reduction(kappa_min, kappa_end, size_reduction_start)
                    sig_off()
                else:
                    raise RuntimeError("LLLReduction object '%s' has no core."%self)
            ELSE:
                raise RuntimeError("LLLReduction object '%s' has no core."%self)
        if not r:
            raise ReductionError( str(get_red_status_str(r)) )

    @property
    def final_kappa(self):
        """FIXME! briefly describe function

        :returns:
        :rtype:

        """
        if self._type == mat_gso_mpz_d:
            return self._core.mpz_d.final_kappa
        IF HAVE_LONG_DOUBLE:
            if self._type == mat_gso_mpz_ld:
                return self._core.mpz_ld.final_kappa
        if self._type == mat_gso_mpz_dpe:
            return self._core.mpz_dpe.final_kappa
        IF HAVE_QD:
            if self._type == mat_gso_mpz_dd:
                return self._core.mpz_dd.final_kappa
            if self._type == mat_gso_mpz_qd:
                return self._core.mpz_qd.final_kappa
        if self._type == mat_gso_mpz_mpfr:
            return self._core.mpz_mpfr.final_kappa

        if self._type == mat_gso_long_d:
            return self._core.long_d.final_kappa
        IF HAVE_LONG_DOUBLE:
            if self._type == mat_gso_long_ld:
                return self._core.long_ld.final_kappa
        if self._type == mat_gso_long_dpe:
            return self._core.long_dpe.final_kappa
        IF HAVE_QD:
            if self._type == mat_gso_long_dd:
                return self._core.long_dd.final_kappa
            if self._type == mat_gso_long_qd:
                return self._core.long_qd.final_kappa
        if self._type == mat_gso_long_mpfr:
            return self._core.long_mpfr.final_kappa

        raise RuntimeError("LLLReduction object '%s' has no core."%self)

    @property
    def last_early_red(self):
        """FIXME! briefly describe function

        :returns:
        :rtype:

        """
        if self._type == mat_gso_mpz_d:
            return self._core.mpz_d.last_early_red
        IF HAVE_LONG_DOUBLE:
            if self._type == mat_gso_mpz_ld:
                return self._core.mpz_ld.last_early_red
        if self._type == mat_gso_mpz_dpe:
            return self._core.mpz_dpe.last_early_red
        IF HAVE_QD:
            if self._type == mat_gso_mpz_dd:
                return self._core.mpz_dd.last_early_red
            if self._type == mat_gso_mpz_qd:
                return self._core.mpz_qd.last_early_red
        if self._type == mat_gso_mpz_mpfr:
            return self._core.mpz_mpfr.last_early_red

        if self._type == mat_gso_long_d:
            return self._core.long_d.last_early_red
        IF HAVE_LONG_DOUBLE:
            if self._type == mat_gso_long_ld:
                return self._core.long_ld.last_early_red
        if self._type == mat_gso_long_dpe:
            return self._core.long_dpe.last_early_red
        IF HAVE_QD:
            if self._type == mat_gso_long_dd:
                return self._core.long_dd.last_early_red
            if self._type == mat_gso_long_qd:
                return self._core.long_qd.last_early_red
        if self._type == mat_gso_long_mpfr:
            return self._core.long_mpfr.last_early_red

        raise RuntimeError("LLLReduction object '%s' has no core."%self)

    @property
    def zeros(self):
        """FIXME! briefly describe function

        :returns:
        :rtype:

        """
        if self._type == mat_gso_mpz_d:
            return self._core.mpz_d.zeros
        IF HAVE_LONG_DOUBLE:
            if self._type == mat_gso_mpz_ld:
                return self._core.mpz_ld.zeros
        if self._type == mat_gso_mpz_dpe:
            return self._core.mpz_dpe.zeros
        IF HAVE_QD:
            if self._type == mat_gso_mpz_dd:
                return self._core.mpz_dd.zeros
            if self._type == mat_gso_mpz_qd:
                return self._core.mpz_qd.zeros
        if self._type == mat_gso_mpz_mpfr:
            return self._core.mpz_mpfr.zeros

        if self._type == mat_gso_long_d:
            return self._core.long_d.zeros
        IF HAVE_LONG_DOUBLE:
            if self._type == mat_gso_long_ld:
                return self._core.long_ld.zeros
        if self._type == mat_gso_long_dpe:
            return self._core.long_dpe.zeros
        IF HAVE_QD:
            if self._type == mat_gso_long_dd:
                return self._core.long_dd.zeros
            if self._type == mat_gso_long_qd:
                return self._core.long_qd.zeros
        if self._type == mat_gso_long_mpfr:
            return self._core.long_mpfr.zeros

        raise RuntimeError("LLLReduction object '%s' has no core."%self)

    @property
    def nswaps(self):
        """FIXME! briefly describe function

        :returns:
        :rtype:

        """
        if self._type == mat_gso_mpz_d:
            return self._core.mpz_d.n_swaps
        IF HAVE_LONG_DOUBLE:
            if self._type == mat_gso_mpz_ld:
                return self._core.mpz_ld.n_swaps
        if self._type == mat_gso_mpz_dpe:
            return self._core.mpz_dpe.n_swaps
        IF HAVE_QD:
            if self._type == mat_gso_mpz_dd:
                return self._core.mpz_dd.n_swaps
            if self._type == mat_gso_mpz_qd:
                return self._core.mpz_qd.n_swaps
        if self._type == mat_gso_mpz_mpfr:
            return self._core.mpz_mpfr.n_swaps

        if self._type == mat_gso_long_d:
            return self._core.long_d.n_swaps
        IF HAVE_LONG_DOUBLE:
            if self._type == mat_gso_long_ld:
                return self._core.long_ld.n_swaps
        if self._type == mat_gso_long_dpe:
            return self._core.long_dpe.n_swaps
        IF HAVE_QD:
            if self._type == mat_gso_long_dd:
                return self._core.long_dd.n_swaps
            if self._type == mat_gso_long_qd:
                return self._core.long_qd.n_swaps
        if self._type == mat_gso_long_mpfr:
            return self._core.long_mpfr.n_swaps

        raise RuntimeError("LLLReduction object '%s' has no core."%self)

    @property
    def delta(self):
        return self._delta

    @property
    def eta(self):
        return self._eta


def lll_reduction(IntegerMatrix B, U=None,
                  double delta=LLL_DEF_DELTA, double eta=LLL_DEF_ETA,
                  method=None, float_type=None,
                  int precision=0, int flags=LLL_DEFAULT):
    u"""Run LLL reduction.

    :param IntegerMatrix B: Integer matrix, modified in place.
    :param U: Transformation matrix or ``None``
    :param double delta: LLL parameter `0.25 < δ ≤ 1`
    :param double eta:  LLL parameter `0 ≤ η < √δ`
    :param method: one of ``'wrapper'``, ``'proved'``, ``'heuristic'``, ``'fast'`` or ``None``.
    :param float_type: an element of `fpylll.config.float_types` or ``None``
    :param precision: bit precision to use if ``float_tpe`` is ``'mpfr'``
    :param int flags: LLL flags.

    :returns: modified matrix ``B``
    """

    cdef IntegerMatrix B_
    if B._type == ZT_MPZ:
        B_ = B
    else:
        B_ = IntegerMatrix(B, int_type="mpz")

    check_delta(delta)
    check_eta(eta)
    check_precision(precision)

    cdef LLLMethod method_
    if method == "wrapper" or method is None:
        method_ = LM_WRAPPER
    elif method == "proved":
        method_ = LM_PROVED
    elif method == "heuristic":
        method_ = LM_HEURISTIC
    elif method == "fast":
        method_ = LM_FAST
    else:
        raise ValueError("Method '%s' unknown."%method)

    if float_type is None and method_ == LM_FAST:
        float_type = "double"

    if method_ == LM_WRAPPER and check_float_type(float_type) != FT_DEFAULT:
        raise ValueError("LLL wrapper function requires float_type==None")

    if method_ == LM_FAST and \
       check_float_type(float_type) not in (FT_DOUBLE, FT_LONG_DOUBLE, FT_DD, FT_QD):
        raise ValueError("LLL fast function requires "
                         "float_type == 'double', 'long double', 'dd' or 'qd'")

    cdef int r
    cdef FloatType ft = check_float_type(float_type)

    if U is not None and isinstance(U, IntegerMatrix):
        sig_on()
        r = lll_reduction_c(B_._core.mpz[0], (<IntegerMatrix>U)._core.mpz[0],
                            delta, eta, method_, ft, precision, flags)
        sig_off()

    else:
        sig_on()
        r = lll_reduction_c(B_._core.mpz[0],
                            delta, eta, method_,
                            ft, precision, flags)
        sig_off()

    if r:
        raise ReductionError( str(get_red_status_str(r)) )

    if B != B_:
        B.set_matrix(B_)
    return B

def is_LLL_reduced(M, delta=LLL_DEF_DELTA, eta=LLL_DEF_ETA):
    """Test if ``M`` is LLL reduced.

    :param M: either an GSO object of an integer matrix or an integer matrix.
    :param delta: LLL parameter δ < 1
    :param eta: LLL parameter η > 0.5

    :returns: Return ``True`` if ``M`` is definitely LLL reduced, ``False`` otherwise.

    Random matrices are typically not LLL reduced::

        >>> from fpylll import IntegerMatrix, LLL
        >>> A = IntegerMatrix(40, 40)
        >>> A.randomize('uniform', bits=32)
        >>> LLL.is_reduced(A)
        False

    LLL reduction should produce matrices which are LLL reduced::

        >>> LLL.reduction(A) # doctest: +ELLIPSIS
        <IntegerMatrix(40, 40) at 0x...>
        >>> LLL.is_reduced(A)
        True

    ..  note:: This function may return ``False`` for LLL reduced matrices if the precision used
        to compute the GSO is too small.
    """
    check_delta(delta)
    check_eta(eta)

    cdef MatGSO M_

    if isinstance(M, MatGSO):
        M_ = M
    elif isinstance(M, IntegerMatrix):
        M_ = MatGSO(M)
        M_.update_gso()
    else:
        raise TypeError("Type '%s' not understood."%type(M))

    if M_._type == mat_gso_mpz_d:
        return bool(is_lll_reduced[Z_NR[mpz_t], FP_NR[double]](M_._core.mpz_d[0], delta, eta))

    IF HAVE_LONG_DOUBLE:
        if M_._type == mat_gso_mpz_ld:
            return bool(is_lll_reduced[Z_NR[mpz_t], FP_NR[longdouble]](M_._core.mpz_ld[0], delta, eta))
    IF HAVE_QD:
        if M_._type == mat_gso_mpz_dd:
            return bool(is_lll_reduced[Z_NR[mpz_t], FP_NR[dd_t]](M_._core.mpz_dd[0], delta, eta))
        if M_._type == mat_gso_mpz_qd:
            return bool(is_lll_reduced[Z_NR[mpz_t], FP_NR[qd_t]](M_._core.mpz_qd[0], delta, eta))
    if M_._type == mat_gso_mpz_mpfr:
        return bool(is_lll_reduced[Z_NR[mpz_t], FP_NR[mpfr_t]](M_._core.mpz_mpfr[0], delta, eta))

    if M_._type == mat_gso_long_d:
        return bool(is_lll_reduced[Z_NR[long], FP_NR[double]](M_._core.long_d[0], delta, eta))

    IF HAVE_LONG_DOUBLE:
        if M_._type == mat_gso_long_ld:
            return bool(is_lll_reduced[Z_NR[long], FP_NR[longdouble]](M_._core.long_ld[0], delta, eta))
    IF HAVE_QD:
        if M_._type == mat_gso_long_dd:
            return bool(is_lll_reduced[Z_NR[long], FP_NR[dd_t]](M_._core.long_dd[0], delta, eta))
        if M_._type == mat_gso_long_qd:
            return bool(is_lll_reduced[Z_NR[long], FP_NR[qd_t]](M_._core.long_qd[0], delta, eta))
    if M_._type == mat_gso_long_mpfr:
        return bool(is_lll_reduced[Z_NR[long], FP_NR[mpfr_t]](M_._core.long_mpfr[0], delta, eta))

    raise RuntimeError("MatGSO object '%s' has no core."%M)


class LLL:
    DEFAULT = LLL_DEFAULT
    VERBOSE = LLL_VERBOSE
    EARLY_RED = LLL_EARLY_RED
    SIEGEL = LLL_SIEGEL

    DEFAULT_DELTA = LLL_DEF_DELTA
    DEFAULT_ETA = LLL_DEF_ETA

    Reduction = LLLReduction

    reduction = staticmethod(lll_reduction)
    is_reduced = staticmethod(is_LLL_reduced)

    Wrapper = Wrapper
