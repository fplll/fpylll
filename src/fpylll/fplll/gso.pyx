# -*- coding: utf-8 -*-
include "fpylll/config.pxi"
include "cysignals/signals.pxi"

"""
Elementary basis operations, Gram matrix and Gram-Schmidt orthogonalization.

..  moduleauthor:: Martin R.  Albrecht <martinralbrecht+fpylll@googlemail.com>
"""


from decl cimport mpz_double, mpz_ld, mpz_dpe, mpz_mpfr, fp_nr_t
from fplll cimport FT_DOUBLE, FT_LONG_DOUBLE, FT_DPE, FT_MPFR, FloatType
from fplll cimport GSO_DEFAULT
from fplll cimport GSO_INT_GRAM
from fplll cimport GSO_OP_FORCE_LONG
from fplll cimport GSO_ROW_EXPO
from fplll cimport MatGSO as MatGSO_c, Z_NR, FP_NR, Matrix
from fplll cimport dpe_t
from fplll cimport get_current_slope
from fpylll.gmp.mpz cimport mpz_t
from fpylll.mpfr.mpfr cimport mpfr_t
from fpylll.util cimport preprocess_indices, check_float_type
from integer_matrix cimport IntegerMatrix

IF HAVE_QD:
    from fpylll.qd.qd cimport dd_real, qd_real
    from decl cimport mpz_dd, mpz_qd
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
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        """
        Exit context for working on rows.

        :param exception_type:
        :param exception_value:
        :param exception_traceback:

        """
        self.M.row_op_end(self.i, self.j)
        return False


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
            return bool(self._core.mpz_double.enable_int_gram)
        if self._type == mpz_ld:
            return bool(self._core.mpz_ld.enable_int_gram)
        if self._type == mpz_dpe:
            return bool(self._core.mpz_dpe.enable_int_gram)
        IF HAVE_QD:
            if self._type == mpz_dd:
                return bool(self._core.mpz_dd.enable_int_gram)
            if self._type == mpz_qd:
                return bool(self._core.mpz_qd.enable_int_gram)
        if self._type == mpz_mpfr:
            return bool(self._core.mpz_mpfr.enable_int_gram)

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
            return bool(self._core.mpz_double.enable_row_expo)
        if self._type == mpz_ld:
            return bool(self._core.mpz_ld.enable_row_expo)
        if self._type == mpz_dpe:
            return bool(self._core.mpz_dpe.enable_row_expo)
        IF HAVE_QD:
            if self._type == mpz_dd:
                return bool(self._core.mpz_dd.enable_row_expo)
            if self._type == mpz_qd:
                return bool(self._core.mpz_qd.enable_row_expo)
        if self._type == mpz_mpfr:
            return bool(self._core.mpz_mpfr.enable_row_expo)

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
            return bool(self._core.mpz_double.enable_transform)
        if self._type == mpz_ld:
            return bool(self._core.mpz_ld.enable_transform)
        if self._type == mpz_dpe:
            return bool(self._core.mpz_dpe.enable_transform)
        IF HAVE_QD:
            if self._type == mpz_dd:
                return bool(self._core.mpz_dd.enable_transform)
            if self._type == mpz_qd:
                return bool(self._core.mpz_qd.enable_transform)
        if self._type == mpz_mpfr:
            return bool(self._core.mpz_mpfr.enable_transform)

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
            return bool(self._core.mpz_double.enable_inverse_transform)
        if self._type == mpz_ld:
            return bool(self._core.mpz_ld.enable_inverse_transform)
        if self._type == mpz_dpe:
            return bool(self._core.mpz_dpe.enable_inverse_transform)
        IF HAVE_QD:
            if self._type == mpz_dd:
                return bool(self._core.mpz_dd.enable_inverse_transform)
            if self._type == mpz_qd:
                return bool(self._core.mpz_qd.enable_inverse_transform)
        if self._type == mpz_mpfr:
            return bool(self._core.mpz_mpfr.enable_inverse_transform)

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
            return bool(self._core.mpz_double.row_op_force_long)
        if self._type == mpz_ld:
            return bool(self._core.mpz_ld.row_op_force_long)
        if self._type == mpz_dpe:
            return bool(self._core.mpz_dpe.row_op_force_long)
        IF HAVE_QD:
            if self._type == mpz_dd:
                return bool(self._core.mpz_dd.row_op_force_long)
            if self._type == mpz_qd:
                return bool(self._core.mpz_qd.row_op_force_long)
        if self._type == mpz_mpfr:
            return bool(self._core.mpz_mpfr.row_op_force_long)

        raise RuntimeError("MatGSO object '%s' has no core."%self)

    def row_op_begin(self, int first, int last):
        """
        Must be called before a sequence of ``row_addmul``.

        :param int first: start index for ``row_addmul`` operations.
        :param int last: final index (exclusive).

        .. note:: It is preferable to use ``MatGSORowOpContext`` via ``row_ops``.
        """
        if self._type == mpz_double:
            return self._core.mpz_double.row_op_begin(first, last)
        if self._type == mpz_ld:
            return self._core.mpz_ld.row_op_begin(first, last)
        if self._type == mpz_dpe:
            return self._core.mpz_dpe.row_op_begin(first, last)
        IF HAVE_QD:
            if self._type == mpz_dd:
                return self._core.mpz_dd.row_op_begin(first, last)
            if self._type == mpz_qd:
                return self._core.mpz_qd.row_op_begin(first, last)
        if self._type == mpz_mpfr:
            return self._core.mpz_mpfr.row_op_begin(first, last)

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
            return self._core.mpz_double.row_op_end(first, last)
        if self._type == mpz_ld:
            return self._core.mpz_ld.row_op_end(first, last)
        if self._type == mpz_dpe:
            return self._core.mpz_dpe.row_op_end(first, last)
        IF HAVE_QD:
            if self._type == mpz_dd:
                return self._core.mpz_dd.row_op_end(first, last)
            if self._type == mpz_qd:
                return self._core.mpz_qd.row_op_end(first, last)
        if self._type == mpz_mpfr:
            return self._core.mpz_mpfr.row_op_end(first, last)

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

        cdef fp_nr_t t

        # TODO: don't just return doubles
        if self._type == mpz_double:
            return self._core.mpz_double.get_gram(t.double, i, j).get_d()
        if self._type == mpz_ld:
            return self._core.mpz_ld.get_gram(t.ld, i, j).get_d()
        if self._type == mpz_dpe:
            return self._core.mpz_dpe.get_gram(t.dpe, i, j).get_d()
        IF HAVE_QD:
            if self._type == mpz_dd:
                return self._core.mpz_dd.get_gram(t.dd, i, j).get_d()
            if self._type == mpz_qd:
                return self._core.mpz_qd.get_gram(t.qd, i, j).get_d()
        if self._type == mpz_mpfr:
            return self._core.mpz_mpfr.get_gram(t.mpfr, i, j).get_d()

        raise RuntimeError("MatGSO object '%s' has no core."%self)

    def get_r(self, int i, int j):
        """
        Return `⟨b_i, b*_j⟩`.

        :param i:
        :param j:

        >>> from fpylll import *
        >>> A = IntegerMatrix.random(5, "uniform", bits=5)
        >>> M = GSO.Mat(A)
        >>> M.update_gso()
        True
        >>> M.get_r(1, 0)
        833.0

        """
        preprocess_indices(i, j, self.d, self.d)
        cdef fp_nr_t t

        # TODO: don't just return doubles
        if self._type == mpz_double:
            return self._core.mpz_double.get_r(t.double, i, j).get_d()
        if self._type == mpz_ld:
            return self._core.mpz_ld.get_r(t.ld, i, j).get_d()
        if self._type == mpz_dpe:
            return self._core.mpz_dpe.get_r(t.dpe, i, j).get_d()
        IF HAVE_QD:
            if self._type == mpz_dd:
                return self._core.mpz_dd.get_r(t.dd, i, j).get_d()
            if self._type == mpz_qd:
                return self._core.mpz_qd.get_r(t.qd, i, j).get_d()
        if self._type == mpz_mpfr:
            return self._core.mpz_mpfr.get_r(t.mpfr, i, j).get_d()

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
            r = self._core.mpz_double.get_r_exp(i, j, expo).get_data()
            return r, expo
        if self._type == mpz_ld:
            r = self._core.mpz_ld.get_r_exp(i, j, expo).get_d()
            return r, expo
        if self._type == mpz_dpe:
            r = self._core.mpz_dpe.get_r_exp(i, j, expo).get_d()
            return r, expo
        IF HAVE_QD:
            if self._type == mpz_dd:
                r = self._core.mpz_dd.get_r_exp(i, j, expo).get_d()
                return r, expo
            if self._type == mpz_qd:
                r = self._core.mpz_qd.get_r_exp(i, j, expo).get_d()
                return r, expo
        if self._type == mpz_mpfr:
            r = self._core.mpz_mpfr.get_r_exp(i, j, expo).get_d()
            return r, expo

        raise RuntimeError("MatGSO object '%s' has no core."%self)


    def get_mu(self, int i, int j):
        """
        Return `<b_i, b^*_j> / ||b^*_j||^2`.

        :param i:
        :param j:

        """
        preprocess_indices(i, j, self.d, self.d)
        cdef fp_nr_t t

        # TODO: don't just return doubles
        if self._type == mpz_double:
            return self._core.mpz_double.get_mu(t.double, i, j).get_d()
        if self._type == mpz_ld:
            return self._core.mpz_ld.get_mu(t.ld, i, j).get_d()
        if self._type == mpz_dpe:
            return self._core.mpz_dpe.get_mu(t.dpe, i, j).get_d()
        IF HAVE_QD:
            if self._type == mpz_dd:
                return self._core.mpz_dd.get_mu(t.dd, i, j).get_d()
            if self._type == mpz_qd:
                return self._core.mpz_qd.get_mu(t.qd, i, j).get_d()
        if self._type == mpz_mpfr:
            return self._core.mpz_mpfr.get_mu(t.mpfr, i, j).get_d()

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
            r = self._core.mpz_double.get_mu_exp(i, j, expo).get_data()
            return r, expo
        if self._type == mpz_ld:
            r = self._core.mpz_ld.get_mu_exp(i, j, expo).get_d()
            return r, expo
        if self._type == mpz_dpe:
            r = self._core.mpz_dpe.get_mu_exp(i, j, expo).get_d()
            return r, expo
        IF HAVE_QD:
            if self._type == mpz_dd:
                r = self._core.mpz_dd.get_mu_exp(i, j, expo).get_d()
                return r, expo
            if self._type == mpz_qd:
                r = self._core.mpz_qd.get_mu_exp(i, j, expo).get_d()
                return r, expo
        if self._type == mpz_mpfr:
            r = self._core.mpz_mpfr.get_mu_exp(i, j, expo).get_d()
            return r, expo

        raise RuntimeError("MatGSO object '%s' has no core."%self)


    def update_gso(self):
        """
        Updates all GSO coefficients (`μ` and `r`).
        """
        cdef int r
        if self._type == mpz_double:
            with nogil:
                r = self._core.mpz_double.update_gso()
            return bool(r)
        if self._type == mpz_ld:
            with nogil:
                r = self._core.mpz_ld.update_gso()
            return bool(r)
        if self._type == mpz_dpe:
            with nogil:
                r = self._core.mpz_dpe.update_gso()
            return bool(r)
        IF HAVE_QD:
            if self._type == mpz_dd:
                with nogil:
                    r = self._core.mpz_dd.update_gso()
                return bool(r)
            if self._type == mpz_qd:
                with nogil:
                    r = self._core.mpz_qd.update_gso()
                return bool(r)
        if self._type == mpz_mpfr:
            with nogil:
                r = self._core.mpz_mpfr.update_gso()
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
            return bool(self._core.mpz_double.update_gso_row(i, last_j))
        if self._type == mpz_ld:
            return bool(self._core.mpz_ld.update_gso_row(i, last_j))
        if self._type == mpz_dpe:
            return bool(self._core.mpz_dpe.update_gso_row(i, last_j))
        IF HAVE_QD:
            if self._type == mpz_dd:
                return bool(self._core.mpz_dd.update_gso_row(i, last_j))
            if self._type == mpz_qd:
                return bool(self._core.mpz_qd.update_gso_row(i, last_j))
        if self._type == mpz_mpfr:
            return bool(self._core.mpz_mpfr.update_gso_row(i, last_j))

        raise RuntimeError("MatGSO object '%s' has no core."%self)


    def discover_all_rows(self):
        """
        Allows ``row_addmul`` for all rows even if the GSO has never been computed.
        """
        if self._type == mpz_double:
            return self._core.mpz_double.discover_all_rows()
        if self._type == mpz_ld:
            return self._core.mpz_ld.discover_all_rows()
        if self._type == mpz_dpe:
            return self._core.mpz_dpe.discover_all_rows()
        IF HAVE_QD:
            if self._type == mpz_dd:
                return self._core.mpz_dd.discover_all_rows()
            if self._type == mpz_qd:
                return self._core.mpz_qd.discover_all_rows()
        if self._type == mpz_mpfr:
            return self._core.mpz_mpfr.discover_all_rows()

        raise RuntimeError("MatGSO object '%s' has no core."%self)

    def move_row(self, int old_r, int new_r):
        """
        Row ``old_r`` becomes row ``new_r`` and intermediate rows are shifted.
        If ``new_r < old_r``, then ``old_r`` must be ``< n_known_rows``.

        :param int old_r: row index
        :param int new_r: row index

        """
        preprocess_indices(old_r, new_r, self.d, self.d)
        if self._type == mpz_double:
            return self._core.mpz_double.move_row(old_r, new_r)
        if self._type == mpz_ld:
            return self._core.mpz_ld.move_row(old_r, new_r)
        if self._type == mpz_dpe:
            return self._core.mpz_dpe.move_row(old_r, new_r)
        IF HAVE_QD:
            if self._type == mpz_dd:
                return self._core.mpz_dd.move_row(old_r, new_r)
            if self._type == mpz_qd:
                return self._core.mpz_qd.move_row(old_r, new_r)
        if self._type == mpz_mpfr:
            return self._core.mpz_mpfr.move_row(old_r, new_r)

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
            [ 1 0 0 12 25 25 ]
            [ 0 1 0 25 12 25 ]
            [ 0 0 1 25 25 12 ]
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
            [ 1 0  0  12  25  25 ]
            [ 0 1  0  25  12  25 ]
            [ 0 0 -1 -25 -25 -12 ]
            [ 0 0  0  31   0   0 ]
            [ 0 0  0   0  31   0 ]
            [ 0 0  0   0   0  31 ]

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
        cdef fp_nr_t x_

        if self._type == mpz_double:
            x_.double = float(x)
            return self._core.mpz_double.row_addmul(i, j, x_.double)
        if self._type == mpz_ld:
            x_.ld = float(x)
            return self._core.mpz_ld.row_addmul(i, j, x_.ld)
        if self._type == mpz_dpe:
            x_.dpe = float(x)
            return self._core.mpz_dpe.row_addmul(i, j, x_.dpe)
        IF HAVE_QD:
            if self._type == mpz_dd:
                x_.dd = float(x)
                return self._core.mpz_dd.row_addmul(i, j, x_.dd)
            if self._type == mpz_qd:
                x_.qd = float(x)
                return self._core.mpz_qd.row_addmul(i, j, x_.qd)
        if self._type == mpz_mpfr:
            x_.mpfr = float(x)
            return self._core.mpz_mpfr.row_addmul(i, j, x_.mpfr)

        raise RuntimeError("MatGSO object '%s' has no core."%self)

    def create_row(self):
        """Adds a zero row to ``B`` (and to ``U`` if ``enable_tranform=true``).  One or several
        operations can be performed on this row with ``row_addmul``, then ``row_op_end`` must be
        called.  Do not use if ``inverse_transform_enabled=true``.

        """
        if self.inverse_transform_enabled:
            raise ValueError("create_row is incompatible with ``inverse_transform_enabled``")

        if self._type == mpz_double:
            return self._core.mpz_double.create_row()
        if self._type == mpz_ld:
            return self._core.mpz_ld.create_row()
        if self._type == mpz_dpe:
            return self._core.mpz_dpe.create_row()
        IF HAVE_QD:
            if self._type == mpz_dd:
                return self._core.mpz_dd.create_row()
            if self._type == mpz_qd:
                return self._core.mpz_qd.create_row()
        if self._type == mpz_mpfr:
            return self._core.mpz_mpfr.create_row()

        raise RuntimeError("MatGSO object '%s' has no core."%self)

    def remove_last_row(self):
        """Remove.  the last row of ``B`` (and of ``U`` if ``enable_transform=true``).  Do not use
        if ``inverse_transform_enabled=true``.

        """
        if self.inverse_transform_enabled:
            raise ValueError("remove_last_row is incompatible with ``inverse_transform_enabled``")

        if self._type == mpz_double:
            return self._core.mpz_double.remove_last_row()
        if self._type == mpz_ld:
            return self._core.mpz_ld.remove_last_row()
        if self._type == mpz_dpe:
            return self._core.mpz_dpe.remove_last_row()
        IF HAVE_QD:
            if self._type == mpz_dd:
                return self._core.mpz_dd.remove_last_row()
            if self._type == mpz_qd:
                return self._core.mpz_qd.remove_last_row()
        if self._type == mpz_mpfr:
            return self._core.mpz_mpfr.remove_last_row()

        raise RuntimeError("MatGSO object '%s' has no core."%self)

    def get_current_slope(self, int start_row, int stop_row):
        """
        Finds the slope of the curve fitted to the lengths of the vectors from ``start_row`` to
        ``stop_row``.  The slope gives an indication of the quality of the LLL-reduced basis.

        :param int start_row: start row index
        :param int stop_row: stop row index (exclusive)

        ..  note:: we call ``get_current_slope`` which is declared in bkz.h
        """

        preprocess_indices(start_row, stop_row, self.d, self.d+1)

        if self._type == mpz_double:
            sig_on()
            r = self._core.mpz_double.get_current_slope(start_row, stop_row)
            sig_off()
            return r
        if self._type == mpz_ld:
            sig_on()
            r = self._core.mpz_ld.get_current_slope(start_row, stop_row)
            sig_off()
            return r
        if self._type == mpz_dpe:
            sig_on()
            r = self._core.mpz_dpe.get_current_slope(start_row, stop_row)
            sig_off()
            return r
        IF HAVE_QD:
            if self._type == mpz_dd:
                sig_on()
                r = self._core.mpz_dd.get_current_slope(start_row, stop_row)
                sig_off()
                return r
            if self._type == mpz_qd:
                sig_on()
                r = self._core.mpz_qd.get_current_slope(start_row, stop_row)
                sig_off()
                return r
        if self._type == mpz_mpfr:
            sig_on()
            r = self._core.mpz_mpfr.get_current_slope(start_row, stop_row)
            sig_off()
            return r

        raise RuntimeError("MatGSO object '%s' has no core."%self)

    def get_root_det(self, int start_row, int stop_row):
        """ Return (squared) root determinant of the basis.

        :param int start_row: start row (inclusive)
        :param int stop_row: stop row (exclusive)

        """
        preprocess_indices(start_row, stop_row, self.d, self.d+1)

        if self._type == mpz_double:
            sig_on()
            r = self._core.mpz_double.get_root_det(start_row, stop_row).get_d()
            sig_off()
            return r
        elif self._type == mpz_ld:
            sig_on()
            r = self._core.mpz_ld.get_root_det(start_row, stop_row).get_d()
            sig_off()
            return r
        elif self._type == mpz_dpe:
            sig_on()
            r = self._core.mpz_dpe.get_root_det(start_row, stop_row).get_d()
            sig_off()
            return r
        elif self._type == mpz_mpfr:
            sig_on()
            r = self._core.mpz_mpfr.get_root_det(start_row, stop_row).get_d()
            sig_off()
            return r
        else:
            IF HAVE_QD:
                if self._type == mpz_dd:
                    sig_on()
                    r = self._core.mpz_dd.get_root_det(start_row, stop_row).get_d()
                    sig_off()
                    return r
                elif self._type == mpz_qd:
                    sig_on()
                    r = self._core.mpz_qd.get_root_det(start_row, stop_row).get_d()
                    sig_off()
                    return r
        raise RuntimeError("MatGSO object '%s' has no core."%self)

    def get_log_det(self, int start_row, int stop_row):
        """ Return log of the (squared) determinant of the basis.

        :param int start_row: start row (inclusive)
        :param int stop_row: stop row (exclusive)

        """
        preprocess_indices(start_row, stop_row, self.d, self.d+1)

        if self._type == mpz_double:
            sig_on()
            r = self._core.mpz_double.get_log_det(start_row, stop_row).get_d()
            sig_off()
            return r
        elif self._type == mpz_ld:
            sig_on()
            r = self._core.mpz_ld.get_log_det(start_row, stop_row).get_d()
            sig_off()
            return r
        elif self._type == mpz_dpe:
            sig_on()
            r = self._core.mpz_dpe.get_log_det(start_row, stop_row).get_d()
            sig_off()
            return r
        elif self._type == mpz_mpfr:
            sig_on()
            r = self._core.mpz_mpfr.get_log_det(start_row, stop_row).get_d()
            sig_off()
            return r
        else:
            IF HAVE_QD:
                if self._type == mpz_dd:
                    sig_on()
                    r = self._core.mpz_dd.get_log_det(start_row, stop_row).get_d()
                    sig_off()
                    return r
                elif self._type == mpz_qd:
                    sig_on()
                    r = self._core.mpz_qd.get_log_det(start_row, stop_row).get_d()
                    sig_off()
                    return r
        raise RuntimeError("MatGSO object '%s' has no core."%self)

    def get_slide_potential(self, int start_row, int stop_row, int block_size):
        """ Return slide potential of the basis

        :param int start_row: start row (inclusive)
        :param int stop_row: stop row (exclusive)
        :param int block_size: block size

        """
        preprocess_indices(start_row, stop_row, self.d, self.d+1)

        if self._type == mpz_double:
            sig_on()
            r = self._core.mpz_double.get_slide_potential(start_row, stop_row, block_size).get_d()
            sig_off()
            return r
        elif self._type == mpz_ld:
            sig_on()
            r = self._core.mpz_ld.get_slide_potential(start_row, stop_row, block_size).get_d()
            sig_off()
            return r
        elif self._type == mpz_dpe:
            sig_on()
            r = self._core.mpz_dpe.get_slide_potential(start_row, stop_row, block_size).get_d()
            sig_off()
            return r
        elif self._type == mpz_mpfr:
            sig_on()
            r = self._core.mpz_mpfr.get_slide_potential(start_row, stop_row, block_size).get_d()
            sig_off()
            return r
        else:
            IF HAVE_QD:
                if self._type == mpz_dd:
                    sig_on()
                    r = self._core.mpz_dd.get_slide_potential(start_row, stop_row, block_size).get_d()
                    sig_off()
                    return r
                elif self._type == mpz_qd:
                    sig_on()
                    r = self._core.mpz_qd.get_slide_potential(start_row, stop_row, block_size).get_d()
                    sig_off()
                    return r
        raise RuntimeError("MatGSO object '%s' has no core."%self)


class GSO:
    DEFAULT=GSO_DEFAULT
    INT_GRAM=GSO_INT_GRAM
    ROW_EXPO=GSO_ROW_EXPO
    OP_FORCE_LONG=GSO_OP_FORCE_LONG
    Mat = MatGSO
