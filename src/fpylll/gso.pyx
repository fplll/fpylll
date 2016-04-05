# -*- coding: utf-8 -*-
include "config.pxi"
include "cysignals/signals.pxi"

"""
Elementary basis operations, Gram matrix and Gram-Schmidt orthogonalization.

..  moduleauthor:: Martin R.  Albrecht <martinralbrecht+fpylll@googlemail.com>
"""


from gmp.mpz cimport mpz_t
from qd.qd cimport dd_real, qd_real
from mpfr.mpfr cimport mpfr_t
from integer_matrix cimport IntegerMatrix
from fplll cimport dpe_t
from fplll cimport MatGSO as MatGSO_c, Z_NR, FP_NR, Matrix
from fplll cimport GSO_DEFAULT
from fplll cimport GSO_INT_GRAM
from fplll cimport GSO_ROW_EXPO
from fplll cimport GSO_OP_FORCE_LONG
from fplll cimport getCurrentSlope, computeGaussHeurDist
from util cimport preprocess_indices, check_float_type
from fpylll cimport mpz_double, mpz_ld, mpz_dpe, mpz_mpfr
from fplll cimport FT_DOUBLE, FT_LONG_DOUBLE, FT_DPE, FT_MPFR, FloatType

IF HAVE_QD:
    from fpylll cimport mpz_dd, mpz_qd
    from fplll cimport FT_DD, FT_QD


class MatGSORowOpContext(object):
    """
    A context in which performing row operations is safe.  When the context is left, the appropriate
    updates are performed by calling ``row_op_end()``.
    """
    def __init__(self, M, i, j):
        """Construct new context for ``M[i:j]``.

        :param M: MatGSO object
        :param i: start row
        :param j: stop row

        """
        self.i = i
        self.j = j
        self.M = M

    def __enter__(self):
        """
        Enter context for working on rows.

        """
        self.M.row_op_begin(self.i, self.j)

    def __exit__(self, exception_type, exception_value, exception_traceback):
        """
        Exit context for working on rows.

        :param exception_type:
        :param exception_value:
        :param exception_traceback:

        """
        self.M.row_op_end(self.i, self.j)


cdef class MatGSO:
    """
    MatGSO provides an interface for performing elementary operations on a basis and computing its
    Gram matrix and its Gram-Schmidt orthogonalization.  The Gram-Schmidt coefficients are computed
    on demand.  The object keeps track of which coefficients are valid after each row operation.
    """

    def __init__(self, IntegerMatrix B, U=None, UinvT=None,
                 int flags=GSO_DEFAULT, float_type="double"):
        """
        :param IntegerMatrix B: The matrix on which row operations are performed.  It must not be
            empty.
        :param IntegerMatrix U: If ``U`` is not empty, operations on ``B`` are also done on ``u``
            (in this case both must have the same number of rows).  If ``u`` is initially the
            identity matrix, multiplying transform by the initial basis gives the current basis.
        :param IntegerMatrix UinvT: Inverse transform (should be empty, which disables the
            computation, or initialized with identity matrix).  It works only if ``U`` is not empty.
        :param int flags: Flags

                - ``GSO.INT_GRAM`` - If true, coefficients of the Gram matrix are computed with
                  exact integer arithmetic.  Otherwise, they are computed in floating-point.  Note
                  that when exact arithmetic is used, all coefficients of the first ``n_known_rows``
                  are continuously updated, whereas in floating-point, they are computed only
                  on-demand.  This option cannot be enabled when ``GSO.ROW_EXPO`` is set.

                - ``GSO.ROW_EXPO`` - If true, each row of ``B`` is normalized by a power of 2 before
                  doing conversion to floating-point, which hopefully avoids some overflows.  This
                  option cannot be enabled if ``GSO.INT_GRAM`` is set and works only with
                  ``float_type="double"`` and ``float_type="long double"``.  It is useless and
                  **must not** be used for ``float_type="dpe"``, ``float_type="dd"``,
                  ``float_type="qd"`` or ``float_type=mpfr_t``.

                - ``GSO.OP_FORCE_LONG`` - Affects the behaviour of ``row_addmul``.  See its
                  documentation.

        :param float_type: A floating point type, i.e. an element of ``fpylll.fpylll.float_types``.

        ..  note:: If ``float_type="mpfr"`` set precision with ``set_precision()`` before
            constructing this object and do not change the precision during the lifetime of this
            object.
        """

        if U is None:
            self.U = IntegerMatrix(0, 0)
        elif isinstance(U, IntegerMatrix):
            if U.nrows != B.nrows:
                raise ValueError("U.nrows != B.nrows")
            self.U = U

        if UinvT is None:
            self.UinvT = IntegerMatrix(0, 0)
        elif isinstance(UinvT, IntegerMatrix):
            if U is None:
                raise ValueError("Uinvt != None but U != None.")
            if UinvT.nrows != B.nrows:
                raise ValueError("UinvT.nrows != B.nrows")
            self.UinvT = UinvT

        cdef Matrix[Z_NR[mpz_t]] *b = <Matrix[Z_NR[mpz_t]]*>B._core
        cdef Matrix[Z_NR[mpz_t]] *u = <Matrix[Z_NR[mpz_t]]*>self.U._core
        cdef Matrix[Z_NR[mpz_t]] *u_inv_t = <Matrix[Z_NR[mpz_t]]*>self.UinvT._core

        cdef FloatType float_type_ = check_float_type(float_type)

        if float_type_ == FT_DOUBLE:
            self._type = mpz_double
            self._core.mpz_double = new MatGSO_c[Z_NR[mpz_t],FP_NR[double]](b[0], u[0], u_inv_t[0], flags)
        elif float_type_ == FT_LONG_DOUBLE:
            self._type = mpz_ld
            self._core.mpz_ld = new MatGSO_c[Z_NR[mpz_t],FP_NR[longdouble]](b[0], u[0], u_inv_t[0], flags)
        elif float_type_ == FT_DPE:
            self._type = mpz_dpe
            self._core.mpz_dpe = new MatGSO_c[Z_NR[mpz_t],FP_NR[dpe_t]](b[0], u[0], u_inv_t[0], flags)
        elif float_type_ == FT_MPFR:
            self._type = mpz_mpfr
            self._core.mpz_mpfr = new MatGSO_c[Z_NR[mpz_t],FP_NR[mpfr_t]](b[0], u[0], u_inv_t[0], flags)
        else:
            IF HAVE_QD:
                if float_type_ == FT_DD:
                    self._type = mpz_dd
                    self._core.mpz_dd = new MatGSO_c[Z_NR[mpz_t],FP_NR[dd_real]](b[0], u[0], u_inv_t[0], flags)
                elif float_type_ == FT_QD:
                    self._type = mpz_qd
                    self._core.mpz_qd = new MatGSO_c[Z_NR[mpz_t],FP_NR[qd_real]](b[0], u[0], u_inv_t[0], flags)
                else:
                    raise ValueError("Float type '%s' not understood."%float_type)
            ELSE:
                raise ValueError("Float type '%s' not understood."%float_type)

        self.B = B

    def __dealloc__(self):
        if self._type == mpz_double:
            del self._core.mpz_double
        if self._type == mpz_ld:
            del self._core.mpz_ld
        if self._type == mpz_dpe:
            del self._core.mpz_dpe
        IF HAVE_QD:
            if self._type == mpz_dd:
                del self._core.mpz_dd
            if self._type == mpz_qd:
                del self._core.mpz_qd
        if self._type == mpz_mpfr:
            del self._core.mpz_mpfr

    def __reduce__(self):
        """
        Make sure attempts at pickling raise an error until proper pickling is implemented.
        """
        raise NotImplementedError

    @property
    def float_type(self):
        """
        >>> from fpylll import IntegerMatrix, GSO, set_precision
        >>> A = IntegerMatrix(10, 10)
        >>> M = GSO.Mat(A)
        >>> M.float_type
        'double'
        >>> set_precision(100)
        53
        >>> M = GSO.Mat(A, float_type='mpfr')
        >>> M.float_type
        'mpfr'

        """
        if self._type == mpz_double:
            return "double"
        if self._type == mpz_ld:
            return "long double"
        if self._type == mpz_dpe:
            return "dpe"
        IF HAVE_QD:
            if self._type == mpz_dd:
                return "dd"
            if self._type == mpz_qd:
                return "qd"
        if self._type == mpz_mpfr:
            return "mpfr"

    @property
    def d(self):
        """
        Number of rows of ``B`` (dimension of the lattice).

        >>> from fpylll import IntegerMatrix, GSO, set_precision
        >>> A = IntegerMatrix(11, 11)
        >>> M = GSO.Mat(A)
        >>> M.d
        11

        """
        if self._type == mpz_double:
            return self._core.mpz_double.d
        if self._type == mpz_ld:
            return self._core.mpz_ld.d
        if self._type == mpz_dpe:
            return self._core.mpz_dpe.d
        IF HAVE_QD:
            if self._type == mpz_dd:
                return self._core.mpz_dd.d
            if self._type == mpz_qd:
                return self._core.mpz_qd.d
        if self._type == mpz_mpfr:
            return self._core.mpz_mpfr.d

        raise RuntimeError("MatGSO object '%s' has no core."%self)

    @property
    def int_gram_enabled(self):
        """
        Exact computation of dot products.

        >>> from fpylll import IntegerMatrix, GSO, set_precision
        >>> A = IntegerMatrix(11, 11)
        >>> M = GSO.Mat(A)
        >>> M.int_gram_enabled
        False

        >>> M = GSO.Mat(A, flags=GSO.INT_GRAM)
        >>> M.int_gram_enabled
        True

        """
        if self._type == mpz_double:
            return bool(self._core.mpz_double.enableIntGram)
        if self._type == mpz_ld:
            return bool(self._core.mpz_ld.enableIntGram)
        if self._type == mpz_dpe:
            return bool(self._core.mpz_dpe.enableIntGram)
        IF HAVE_QD:
            if self._type == mpz_dd:
                return bool(self._core.mpz_dd.enableIntGram)
            if self._type == mpz_qd:
                return bool(self._core.mpz_qd.enableIntGram)
        if self._type == mpz_mpfr:
            return bool(self._core.mpz_mpfr.enableIntGram)

        raise RuntimeError("MatGSO object '%s' has no core."%self)

    @property
    def row_expo_enabled(self):
        """
        Normalization of each row of b by a power of 2.

        >>> from fpylll import IntegerMatrix, GSO, set_precision
        >>> A = IntegerMatrix(11, 11)
        >>> M = GSO.Mat(A)
        >>> M.row_expo_enabled
        False

        >>> M = GSO.Mat(A, flags=GSO.ROW_EXPO)
        >>> M.row_expo_enabled
        True

        """
        if self._type == mpz_double:
            return bool(self._core.mpz_double.enableRowExpo)
        if self._type == mpz_ld:
            return bool(self._core.mpz_ld.enableRowExpo)
        if self._type == mpz_dpe:
            return bool(self._core.mpz_dpe.enableRowExpo)
        IF HAVE_QD:
            if self._type == mpz_dd:
                return bool(self._core.mpz_dd.enableRowExpo)
            if self._type == mpz_qd:
                return bool(self._core.mpz_qd.enableRowExpo)
        if self._type == mpz_mpfr:
            return bool(self._core.mpz_mpfr.enableRowExpo)

        raise RuntimeError("MatGSO object '%s' has no core."%self)

    @property
    def transform_enabled(self):
        """
        Computation of the transform matrix.

        >>> from fpylll import IntegerMatrix, GSO, set_precision
        >>> A = IntegerMatrix(11, 11)
        >>> M = GSO.Mat(A)
        >>> M.transform_enabled
        False

        >>> U = IntegerMatrix.identity(11)
        >>> M = GSO.Mat(A, U=U)

        >>> M.transform_enabled
        True

        """
        if self._type == mpz_double:
            return bool(self._core.mpz_double.enableTransform)
        if self._type == mpz_ld:
            return bool(self._core.mpz_ld.enableTransform)
        if self._type == mpz_dpe:
            return bool(self._core.mpz_dpe.enableTransform)
        IF HAVE_QD:
            if self._type == mpz_dd:
                return bool(self._core.mpz_dd.enableTransform)
            if self._type == mpz_qd:
                return bool(self._core.mpz_qd.enableTransform)
        if self._type == mpz_mpfr:
            return bool(self._core.mpz_mpfr.enableTransform)

        raise RuntimeError("MatGSO object '%s' has no core."%self)

    @property
    def inverse_transform_enabled(self):
        """
        Computation of the inverse transform matrix (transposed).

        >>> from fpylll import IntegerMatrix, GSO, set_precision
        >>> A = IntegerMatrix(11, 11)
        >>> M = GSO.Mat(A)
        >>> M.inverse_transform_enabled
        False

        >>> U = IntegerMatrix.identity(11)
        >>> UinvT = IntegerMatrix.identity(11)
        >>> M = GSO.Mat(A, U=U, UinvT=UinvT)
        >>> M.inverse_transform_enabled
        True

        """
        if self._type == mpz_double:
            return bool(self._core.mpz_double.enableInvTransform)
        if self._type == mpz_ld:
            return bool(self._core.mpz_ld.enableInvTransform)
        if self._type == mpz_dpe:
            return bool(self._core.mpz_dpe.enableInvTransform)
        IF HAVE_QD:
            if self._type == mpz_dd:
                return bool(self._core.mpz_dd.enableInvTransform)
            if self._type == mpz_qd:
                return bool(self._core.mpz_qd.enableInvTransform)
        if self._type == mpz_mpfr:
            return bool(self._core.mpz_mpfr.enableInvTransform)

        raise RuntimeError("MatGSO object '%s' has no core."%self)

    @property
    def row_op_force_long(self):
        """
        Changes the behaviour of ``row_addmul``, see its documentation.

        >>> from fpylll import IntegerMatrix, GSO, set_precision
        >>> A = IntegerMatrix(11, 11)
        >>> M = GSO.Mat(A)
        >>> M.row_op_force_long
        False

        >>> M = GSO.Mat(A, flags=GSO.OP_FORCE_LONG)
        >>> M.row_op_force_long
        True

        """
        if self._type == mpz_double:
            return bool(self._core.mpz_double.rowOpForceLong)
        if self._type == mpz_ld:
            return bool(self._core.mpz_ld.rowOpForceLong)
        if self._type == mpz_dpe:
            return bool(self._core.mpz_dpe.rowOpForceLong)
        IF HAVE_QD:
            if self._type == mpz_dd:
                return bool(self._core.mpz_dd.rowOpForceLong)
            if self._type == mpz_qd:
                return bool(self._core.mpz_qd.rowOpForceLong)
        if self._type == mpz_mpfr:
            return bool(self._core.mpz_mpfr.rowOpForceLong)

        raise RuntimeError("MatGSO object '%s' has no core."%self)

    def row_op_begin(self, int first, int last):
        """
        Must be called before a sequence of ``row_addmul``.

        :param int first: start index for ``row_addmul`` operations.
        :param int last: final index (exclusive).

        .. note:: It is preferable to use ``MatGSORowOpContext`` via ``row_ops``.
        """
        if self._type == mpz_double:
            return self._core.mpz_double.rowOpBegin(first, last)
        if self._type == mpz_ld:
            return self._core.mpz_ld.rowOpBegin(first, last)
        if self._type == mpz_dpe:
            return self._core.mpz_dpe.rowOpBegin(first, last)
        IF HAVE_QD:
            if self._type == mpz_dd:
                return self._core.mpz_dd.rowOpBegin(first, last)
            if self._type == mpz_qd:
                return self._core.mpz_qd.rowOpBegin(first, last)
        if self._type == mpz_mpfr:
            return self._core.mpz_mpfr.rowOpBegin(first, last)

        raise RuntimeError("MatGSO object '%s' has no core."%self)

    def row_op_end(self, int first, int last):
        """
        Must be called after a sequence of ``row_addmul``.  This invalidates the `i`-th line of the
        GSO.

        :param int first: start index to invalidate.
        :param int last:  final index to invalidate (exclusive).

        .. note:: It is preferable to use ``MatGSORowOpContext`` via ``row_ops``.
        """
        if self._type == mpz_double:
            return self._core.mpz_double.rowOpEnd(first, last)
        if self._type == mpz_ld:
            return self._core.mpz_ld.rowOpEnd(first, last)
        if self._type == mpz_dpe:
            return self._core.mpz_dpe.rowOpEnd(first, last)
        IF HAVE_QD:
            if self._type == mpz_dd:
                return self._core.mpz_dd.rowOpEnd(first, last)
            if self._type == mpz_qd:
                return self._core.mpz_qd.rowOpEnd(first, last)
        if self._type == mpz_mpfr:
            return self._core.mpz_mpfr.rowOpEnd(first, last)

        raise RuntimeError("MatGSO object '%s' has no core."%self)

    def row_ops(self, int first, int last):
        """Return context in which ``row_addmul`` operations are safe.

        :param int first: start index.
        :param int last: final index (exclusive).

        """
        return MatGSORowOpContext(self, first, last)

    def get_gram(self, int i, int j):
        """
        Return Gram matrix coefficients (0 ≤ i ≤ ``n_known_rows`` and 0 ≤ j ≤ i).  If
        ``enable_row_expo`` is false, returns the dot product `⟨b_i, b_j⟩`.  If ``enable_row_expo``
        is true, returns `⟨b_i, b_j⟩/ 2^{(r_i + r_j)}`, where `r_i` and `r_j` are the row exponents
        of rows `i` and `j` respectively.

        :param int i:
        :param int j:

        """
        preprocess_indices(i, j, self.d, self.d)

        cdef FP_NR[double] t_double
        cdef FP_NR[longdouble] t_ld
        cdef FP_NR[dpe_t] t_dpe
        IF HAVE_QD:
            cdef FP_NR[dd_real] t_dd
            cdef FP_NR[qd_real] t_qd
        cdef FP_NR[mpfr_t] t_mpfr

        # TODO: don't just return doubles
        if self._type == mpz_double:
            self._core.mpz_double.getGram(t_double, i, j)
            return t_double.getData()
        if self._type == mpz_ld:
            self._core.mpz_ld.getGram(t_ld, i, j)
            return t_ld.getData()
        if self._type == mpz_dpe:
            self._core.mpz_dpe.getGram(t_dpe, i, j)
            return t_dpe.get_d()
        IF HAVE_QD:
            if self._type == mpz_dd:
                self._core.mpz_dd.getGram(t_dd, i, j)
                return t_dd.get_d()
            if self._type == mpz_qd:
                self._core.mpz_qd.getGram(t_qd, i, j)
                return t_qd.get_d()
        if self._type == mpz_mpfr:
            self._core.mpz_mpfr.getGram(t_mpfr, i, j)
            return t_mpfr.get_d()

        raise RuntimeError("MatGSO object '%s' has no core."%self)

    def get_r(self, int i, int j):
        """
        Return `⟨b_i, b*_j⟩`.

        :param i:
        :param j:

        """
        preprocess_indices(i, j, self.d, self.d)
        cdef FP_NR[double] t_double
        cdef FP_NR[longdouble] t_ld
        cdef FP_NR[dpe_t] t_dpe
        IF HAVE_QD:
            cdef FP_NR[dd_real] t_dd
            cdef FP_NR[qd_real] t_qd
        cdef FP_NR[mpfr_t] t_mpfr

        # TODO: don't just return doubles
        if self._type == mpz_double:
            self._core.mpz_double.getR(t_double, i, j)
            return t_double.getData()
        if self._type == mpz_ld:
            self._core.mpz_ld.getR(t_ld, i, j)
            return t_ld.get_d()
        if self._type == mpz_dpe:
            self._core.mpz_dpe.getR(t_dpe, i, j)
            return t_dpe.get_d()
        IF HAVE_QD:
            if self._type == mpz_dd:
                self._core.mpz_dd.getR(t_dd, i, j)
                return t_dd.get_d()
            if self._type == mpz_qd:
                self._core.mpz_qd.getR(t_qd, i, j)
                return t_qd.get_d()
        if self._type == mpz_mpfr:
            self._core.mpz_mpfr.getR(t_mpfr, i, j)
            return t_mpfr.get_d()

        raise RuntimeError("MatGSO object '%s' has no core."%self)

    def get_r_exp(self, int i, int j):
        """
        Return `f = r_{i, j}` and exponent `x` such that `⟨b_i, b^*_j⟩ = f ⋅ 2^x`.  If
        ``enable_row_expo`` is false, `x` is always 0.  If ``enable_row_expo`` is true, `x = r_i +
        r_j`, where `r_i` and `r_j` are the row exponents of rows `i` and `j` respectively.

        .. note:: It is assumed that `r(i, j)` is valid.

        :param i:
        :param j:

        """
        preprocess_indices(i, j, self.d, self.d)
        cdef double r = 0.0
        cdef long expo = 0

        # TODO: don't just return doubles
        if self._type == mpz_double:
            r = self._core.mpz_double.getRExp(i, j, expo).getData()
            return r, expo
        if self._type == mpz_ld:
            r = self._core.mpz_ld.getRExp(i, j, expo).get_d()
            return r, expo
        if self._type == mpz_dpe:
            r = self._core.mpz_dpe.getRExp(i, j, expo).get_d()
            return r, expo
        IF HAVE_QD:
            if self._type == mpz_dd:
                r = self._core.mpz_dd.getRExp(i, j, expo).get_d()
                return r, expo
            if self._type == mpz_qd:
                r = self._core.mpz_qd.getRExp(i, j, expo).get_d()
                return r, expo
        if self._type == mpz_mpfr:
            r = self._core.mpz_mpfr.getRExp(i, j, expo).get_d()
            return r, expo

        raise RuntimeError("MatGSO object '%s' has no core."%self)


    def get_mu(self, int i, int j):
        """
        Return `<b_i, b^*_j> / ||b^*_j||^2`.

        :param i:
        :param j:

        """
        preprocess_indices(i, j, self.d, self.d)
        cdef FP_NR[double] t_double
        cdef FP_NR[longdouble] t_ld
        cdef FP_NR[dpe_t] t_dpe
        IF HAVE_QD:
            cdef FP_NR[dd_real] t_dd
            cdef FP_NR[qd_real] t_qd
        cdef FP_NR[mpfr_t] t_mpfr

        # TODO: don't just return doubles
        if self._type == mpz_double:
            self._core.mpz_double.getMu(t_double, i, j)
            return t_double.getData()
        if self._type == mpz_ld:
            self._core.mpz_ld.getMu(t_ld, i, j)
            return t_ld.get_d()
        if self._type == mpz_dpe:
            self._core.mpz_dpe.getMu(t_dpe, i, j)
            return t_dpe.get_d()
        IF HAVE_QD:
            if self._type == mpz_dd:
                self._core.mpz_dd.getMu(t_dd, i, j)
                return t_dd.get_d()
            if self._type == mpz_qd:
                self._core.mpz_qd.getMu(t_qd, i, j)
                return t_qd.get_d()
        if self._type == mpz_mpfr:
            self._core.mpz_mpfr.getMu(t_mpfr, i, j)
            return t_mpfr.get_d()

        raise RuntimeError("MatGSO object '%s' has no core."%self)

    def get_mu_exp(self, int i, int j):
        """
        Return `f = μ_{i, j}` and exponent `x` such that `f ⋅ 2^x = ⟨b_i, b^*_j⟩ / ‖b^*_j‖^2`.  If
        ``enable_row_expo`` is false, `x` is always zero.  If ``enable_row_expo`` is true, `x = r_i
        - r_j`, where `r_i` and `r_j` are the row exponents of rows `i` and `j` respectively.

        .. note:: It is assumed that `μ_{i, j}` is valid.

        :param i:
        :param j:

        """
        preprocess_indices(i, j, self.d, self.d)
        cdef double r = 0.0
        cdef long expo = 0

        # TODO: don't just return doubles
        if self._type == mpz_double:
            r = self._core.mpz_double.getMuExp(i, j, expo).getData()
            return r, expo
        if self._type == mpz_ld:
            r = self._core.mpz_ld.getMuExp(i, j, expo).get_d()
            return r, expo
        if self._type == mpz_dpe:
            r = self._core.mpz_dpe.getMuExp(i, j, expo).get_d()
            return r, expo
        IF HAVE_QD:
            if self._type == mpz_dd:
                r = self._core.mpz_dd.getMuExp(i, j, expo).get_d()
                return r, expo
            if self._type == mpz_qd:
                r = self._core.mpz_qd.getMuExp(i, j, expo).get_d()
                return r, expo
        if self._type == mpz_mpfr:
            r = self._core.mpz_mpfr.getMuExp(i, j, expo).get_d()
            return r, expo

        raise RuntimeError("MatGSO object '%s' has no core."%self)


    def update_gso(self):
        """
        Updates all GSO coefficients (`μ` and `r`).
        """
        cdef int r
        if self._type == mpz_double:
            with nogil:
                r = self._core.mpz_double.updateGSO()
            return bool(r)
        if self._type == mpz_ld:
            with nogil:
                r = self._core.mpz_ld.updateGSO()
            return bool(r)
        if self._type == mpz_dpe:
            with nogil:
                r = self._core.mpz_dpe.updateGSO()
            return bool(r)
        IF HAVE_QD:
            if self._type == mpz_dd:
                with nogil:
                    r = self._core.mpz_dd.updateGSO()
                return bool(r)
            if self._type == mpz_qd:
                with nogil:
                    r = self._core.mpz_qd.updateGSO()
                return bool(r)
        if self._type == mpz_mpfr:
            with nogil:
                r = self._core.mpz_mpfr.updateGSO()
            return bool(r)

        raise RuntimeError("MatGSO object '%s' has no core."%self)

    def update_gso_row(self, int i, int last_j):
        """
        Updates `r_{i, j}` and `μ_{i, j}` if needed for all `j` in `[0, last_j]`.  All coefficients
        of `r` and `μ` above the `i`-th row in columns `[0, min(last_j, i - 1)]` must be valid.

        :param int i:
        :param int last_j:

        """
        if self._type == mpz_double:
            return bool(self._core.mpz_double.updateGSORow(i, last_j))
        if self._type == mpz_ld:
            return bool(self._core.mpz_ld.updateGSORow(i, last_j))
        if self._type == mpz_dpe:
            return bool(self._core.mpz_dpe.updateGSORow(i, last_j))
        IF HAVE_QD:
            if self._type == mpz_dd:
                return bool(self._core.mpz_dd.updateGSORow(i, last_j))
            if self._type == mpz_qd:
                return bool(self._core.mpz_qd.updateGSORow(i, last_j))
        if self._type == mpz_mpfr:
            return bool(self._core.mpz_mpfr.updateGSORow(i, last_j))

        raise RuntimeError("MatGSO object '%s' has no core."%self)


    def discover_all_rows(self):
        """
        Allows ``row_addmul`` for all rows even if the GSO has never been computed.
        """
        if self._type == mpz_double:
            return self._core.mpz_double.discoverAllRows()
        if self._type == mpz_ld:
            return self._core.mpz_ld.discoverAllRows()
        if self._type == mpz_dpe:
            return self._core.mpz_dpe.discoverAllRows()
        IF HAVE_QD:
            if self._type == mpz_dd:
                return self._core.mpz_dd.discoverAllRows()
            if self._type == mpz_qd:
                return self._core.mpz_qd.discoverAllRows()
        if self._type == mpz_mpfr:
            return self._core.mpz_mpfr.discoverAllRows()

        raise RuntimeError("MatGSO object '%s' has no core."%self)

    def move_row(self, int old_r, int new_r):
        """
        Row ``old_r`` becomes row ``new_r`` and intermediate rows are shifted.
        If ``new_r < old_r``, then ``old_r`` must be ``< nKnownRows``.

        :param int old_r: row index
        :param int new_r: row index

        """
        preprocess_indices(old_r, new_r, self.d, self.d)
        if self._type == mpz_double:
            return self._core.mpz_double.moveRow(old_r, new_r)
        if self._type == mpz_ld:
            return self._core.mpz_ld.moveRow(old_r, new_r)
        if self._type == mpz_dpe:
            return self._core.mpz_dpe.moveRow(old_r, new_r)
        IF HAVE_QD:
            if self._type == mpz_dd:
                return self._core.mpz_dd.moveRow(old_r, new_r)
            if self._type == mpz_qd:
                return self._core.mpz_qd.moveRow(old_r, new_r)
        if self._type == mpz_mpfr:
            return self._core.mpz_mpfr.moveRow(old_r, new_r)

        raise RuntimeError("MatGSO object '%s' has no core."%self)

    def negate_row(self, int i):
        """Set `b_i` to `-b_i`.

        :param int i: index of the row to negate

        Example::

            >>> from fpylll import *
            >>> set_random_seed(42)
            >>> A = IntegerMatrix(6, 6)
            >>> A.randomize("ntrulike", bits=6, q=31)
            >>> print(A)
            [ 1 0 0 57 25  7 ]
            [ 0 1 0 25  7 57 ]
            [ 0 0 1  7 57 25 ]
            [ 0 0 0 31  0  0 ]
            [ 0 0 0  0 31  0 ]
            [ 0 0 0  0  0 31 ]
            >>> M = GSO.Mat(A)
            >>> M.update_gso()
            True
            >>> with M.row_ops(2,2):
            ...     M.negate_row(2)
            ...
            >>> print(A)
            [ 1 0  0 57  25   7 ]
            [ 0 1  0 25   7  57 ]
            [ 0 0 -1 -7 -57 -25 ]
            [ 0 0  0 31   0   0 ]
            [ 0 0  0  0  31   0 ]
            [ 0 0  0  0   0  31 ]

        """
        self.row_addmul(i, i, -2.0)


    def row_addmul(self, int i, int j, x):
        """Set `b_i = b_i + x ⋅ b_j`.

        After one or several calls to ``row_addmul``, ``row_op_end`` must be called.

        If ``row_op_force_long=true``, ``x`` is always converted to (``2^expo * long``) instead of
        (``2^expo * ZT``), which is faster if ``ZT=mpz_t`` but might lead to a loss of precision in
        LLL, more Babai iterations are needed.

        :param int i: target row
        :param int j: source row
        :param x: multiplier

        """
        preprocess_indices(i, j, self.d, self.d)
        cdef double x_ = x
        cdef FP_NR[double] x_double
        cdef FP_NR[longdouble] x_ld
        cdef FP_NR[dpe_t] x_dpe
        IF HAVE_QD:
            cdef FP_NR[dd_real] x_dd
            cdef FP_NR[qd_real] x_qd
        cdef FP_NR[mpfr_t] x_mpfr

        if self._type == mpz_double:
            x_double = x_
            return self._core.mpz_double.row_addmul(i, j, x_double)
        if self._type == mpz_ld:
            x_ld = x_
            return self._core.mpz_ld.row_addmul(i, j, x_ld)
        if self._type == mpz_dpe:
            x_dpe = x_
            return self._core.mpz_dpe.row_addmul(i, j, x_dpe)
        IF HAVE_QD:
            if self._type == mpz_dd:
                x_dd = x_
                return self._core.mpz_dd.row_addmul(i, j, x_dd)
            if self._type == mpz_qd:
                x_qd = x_
                return self._core.mpz_qd.row_addmul(i, j, x_qd)
        if self._type == mpz_mpfr:
            x_mpfr = x_
            return self._core.mpz_mpfr.row_addmul(i, j, x_mpfr)

        raise RuntimeError("MatGSO object '%s' has no core."%self)

    def create_row(self):
        """Adds a zero row to ``B`` (and to ``U`` if ``enable_tranform=true``).  One or several
        operations can be performed on this row with ``row_addmul``, then ``row_op_end`` must be
        called.  Do not use if ``inverse_transform_enabled=true``.

        """
        if self.inverse_transform_enabled:
            raise ValueError("create_row is incompatible with ``inverse_transform_enabled``")

        if self._type == mpz_double:
            return self._core.mpz_double.createRow()
        if self._type == mpz_ld:
            return self._core.mpz_ld.createRow()
        if self._type == mpz_dpe:
            return self._core.mpz_dpe.createRow()
        IF HAVE_QD:
            if self._type == mpz_dd:
                return self._core.mpz_dd.createRow()
            if self._type == mpz_qd:
                return self._core.mpz_qd.createRow()
        if self._type == mpz_mpfr:
            return self._core.mpz_mpfr.createRow()

        raise RuntimeError("MatGSO object '%s' has no core."%self)

    def remove_last_row(self):
        """Remove.  the last row of ``B`` (and of ``U`` if ``enable_transform=true``).  Do not use
        if ``inverse_transform_enabled=true``.

        """
        if self.inverse_transform_enabled:
            raise ValueError("remove_last_row is incompatible with ``inverse_transform_enabled``")

        if self._type == mpz_double:
            return self._core.mpz_double.removeLastRow()
        if self._type == mpz_ld:
            return self._core.mpz_ld.removeLastRow()
        if self._type == mpz_dpe:
            return self._core.mpz_dpe.removeLastRow()
        IF HAVE_QD:
            if self._type == mpz_dd:
                return self._core.mpz_dd.removeLastRow()
            if self._type == mpz_qd:
                return self._core.mpz_qd.removeLastRow()
        if self._type == mpz_mpfr:
            return self._core.mpz_mpfr.removeLastRow()

        raise RuntimeError("MatGSO object '%s' has no core."%self)

    def get_current_slope(self, int start_row, int stop_row):
        """
        Finds the slope of the curve fitted to the lengths of the vectors from ``start_row`` to
        ``stop_row``.  The slope gives an indication of the quality of the LLL-reduced basis.

        :param int start_row: start row index
        :param int stop_row: stop row index (exclusive)

        ..  note:: we call ``getCurrentSlope`` which is declared in bkz.h
        """
        if self._type == mpz_double:
            return getCurrentSlope[FP_NR[double]](self._core.mpz_double[0], start_row, stop_row)
        if self._type == mpz_ld:
            return getCurrentSlope[FP_NR[longdouble]](self._core.mpz_ld[0], start_row, stop_row)
        if self._type == mpz_dpe:
            return getCurrentSlope[FP_NR[dpe_t]](self._core.mpz_dpe[0], start_row, stop_row)
        IF HAVE_QD:
            if self._type == mpz_dd:
                return getCurrentSlope[FP_NR[dd_real]](self._core.mpz_dd[0], start_row, stop_row)
            if self._type == mpz_qd:
                return getCurrentSlope[FP_NR[qd_real]](self._core.mpz_qd[0], start_row, stop_row)
        if self._type == mpz_mpfr:
            return getCurrentSlope[FP_NR[mpfr_t]](self._core.mpz_mpfr[0], start_row, stop_row)

        raise RuntimeError("MatGSO object '%s' has no core."%self)


    def compute_gaussian_heuristic_distance(self, int kappa, int block_size,
                                            double max_dist, int max_dist_expo,
                                            double gh_factor):
        """
        Use Gaussian Heuristic Distance to compute a bound on the length of the shortest vector.

        :param int kappa: row index
        :param int block_size: block size
        :param double max_dist: current maximal distance
        :param int max_dist_expo: exponent of current maximal distance
        :param double gh_factor: Gaussian heuristic factor to use

        :returns: (max_dist, max_dist_expo)

        ..  note:: we call ``computeGaussianHeurDist`` which is declared in bkz.h
        """

        cdef FP_NR[double] max_dist_double
        cdef FP_NR[longdouble] max_dist_ld
        cdef FP_NR[dpe_t] max_dist_dpe
        IF HAVE_QD:
            cdef FP_NR[dd_real] max_dist_dd
            cdef FP_NR[qd_real] max_dist_qd
        cdef FP_NR[mpfr_t] max_dist_mpfr

        if self._type == mpz_double:
            max_dist_double = max_dist
            computeGaussHeurDist[FP_NR[double]](self._core.mpz_double[0],
                                                max_dist_double, max_dist_expo, kappa, block_size, gh_factor)
            max_dist = max_dist_double.get_d()
            return max_dist, max_dist_expo
        if self._type == mpz_ld:
            max_dist_ld = max_dist
            computeGaussHeurDist[FP_NR[longdouble]](self._core.mpz_ld[0],
                                                    max_dist_ld, max_dist_expo, kappa, block_size, gh_factor)
            max_dist = max_dist_ld.get_d()
            return max_dist, max_dist_expo
        if self._type == mpz_dpe:
            max_dist_dpe = max_dist
            computeGaussHeurDist[FP_NR[dpe_t]](self._core.mpz_dpe[0],
                                               max_dist_dpe, max_dist_expo, kappa, block_size, gh_factor)
            max_dist = max_dist_dpe.get_d()
            return max_dist, max_dist_expo

        IF HAVE_QD:
            if self._type == mpz_dd:
                max_dist_dd = max_dist
                computeGaussHeurDist[FP_NR[dd_real]](self._core.mpz_dd[0],
                                                     max_dist_dd, max_dist_expo, kappa, block_size, gh_factor)
                max_dist = max_dist_dd.get_d()
                return max_dist, max_dist_expo
            if self._type == mpz_qd:
                max_dist_qd = max_dist
                computeGaussHeurDist[FP_NR[qd_real]](self._core.mpz_qd[0],
                                                     max_dist_qd, max_dist_expo, kappa, block_size, gh_factor)
                max_dist = max_dist_qd.get_d()
                return max_dist, max_dist_expo

        if self._type == mpz_mpfr:
            max_dist_mpfr = max_dist
            computeGaussHeurDist[FP_NR[mpfr_t]](self._core.mpz_mpfr[0],
                                                max_dist_mpfr, max_dist_expo, kappa, block_size, gh_factor)
            max_dist = max_dist_mpfr.get_d()
            return max_dist, max_dist_expo

        raise RuntimeError("MatGSO object '%s' has no core."%self)


class GSO:
    DEFAULT=GSO_DEFAULT
    INT_GRAM=GSO_INT_GRAM
    ROW_EXPO=GSO_ROW_EXPO
    OP_FORCE_LONG=GSO_OP_FORCE_LONG
    Mat = MatGSO
