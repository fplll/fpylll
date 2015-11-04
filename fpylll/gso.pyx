from gmp.mpz cimport mpz_t
from mpfr.mpfr cimport mpfr_t
from integer_matrix cimport IntegerMatrix
from fplll cimport MatGSO as MatGSO_c, Z_NR, FP_NR, Matrix
from fplll cimport GSO_DEFAULT as GSO_DEFAULT_c
from fplll cimport GSO_INT_GRAM as GSO_INT_GRAM_c
from fplll cimport GSO_ROW_EXPO as GSO_ROW_EXPO_c
from fplll cimport GSO_OP_FORCE_LONG as GSO_OP_FORCE_LONG_c
from util cimport preprocess_indices
from fpylll cimport mpz_double, mpz_mpfr

DEFAULT=GSO_DEFAULT_c
INT_GRAM=GSO_INT_GRAM_c
ROW_EXPO=GSO_ROW_EXPO_c
OP_FORCE_LONG=GSO_OP_FORCE_LONG_c


class MatGSORowOpContext(object):
    def __init__(self, MatGSO m, int i, int j):
        """FIXME! briefly describe function

        :param m: MatGSO object
        :param i: start row
        :param j: stop row

        """
        self.i = i
        self.j = j
        self.m = m

    def __enter__(self):
        """
        Enter context for working on rows.

        """
        self.m.row_op_begin(self.i, self.j)

    def __exit__(self, exception_type, exception_value, exception_traceback):
        """
        Exit context for working on rows.

        :param exception_type:
        :param exception_value:
        :param exception_traceback:

        """
        self.m.row_op_end(self.i, self.j)


cdef class MatGSO:
    """
    """
    def __init__(self, IntegerMatrix B, U=None, UinvT=None,
                 int flags=DEFAULT, float_type="double"):
        """FIXME! briefly describe function

        :param IntegerMatrix B:
        :param IntegerMatrix U:
        :param IntegerMatrix UinvT:
        :param int flags:
        :param float_type:
        :returns:
        :rtype:

        """

        if U is None:
            self._U = IntegerMatrix(0, 0)
        elif isinstance(U, IntegerMatrix):
            self._U = U
            self._U.gen_identity(B.nrows)

        if UinvT is None:
            self._UinvT = IntegerMatrix(0, 0)
        elif isinstance(UinvT, IntegerMatrix):
            self._UinvT = UinvT
            self._UinvT.gen_identity(B.nrows)

        cdef Matrix[Z_NR[mpz_t]] *b = <Matrix[Z_NR[mpz_t]]*>B._core
        cdef Matrix[Z_NR[mpz_t]] *u = <Matrix[Z_NR[mpz_t]]*>self._U._core
        cdef Matrix[Z_NR[mpz_t]] *u_inv_t = <Matrix[Z_NR[mpz_t]]*>self._UinvT._core

        if float_type == "double":
            self._type = mpz_double
            self._core.mpz_double = new MatGSO_c[Z_NR[mpz_t],FP_NR[double]](b[0], u[0], u_inv_t[0], flags)
        elif float_type == "mpfr":
            self._type = mpz_mpfr
            self._core.mpz_mpfr = new MatGSO_c[Z_NR[mpz_t],FP_NR[mpfr_t]](b[0], u[0], u_inv_t[0], flags)
        else:
            raise ValueError("Float type '%s' not understood."%float_type)

        self._B = B

    def __dealloc__(self):
        if self._type == mpz_double:
            del self._core.mpz_double
        if self._type == mpz_mpfr:
            del self._core.mpz_mpfr

    @property
    def d(self):
        if self._type == mpz_double:
            return self._core.mpz_double.d
        elif self._type == mpz_mpfr:
            return self._core.mpz_mpfr.d
        else:
            raise RuntimeError("MatGSO object '%s' has no core."%self)

    @property
    def int_gram_enabled(self):
        if self._type == mpz_double:
            return self._core.mpz_double.enableIntGram
        elif self._type == mpz_mpfr:
            return self._core.mpz_mpfr.enableIntGram
        else:
            raise RuntimeError("MatGSO object '%s' has no core."%self)

    @property
    def row_expo_enabled(self):
        if self._type == mpz_double:
            return self._core.mpz_double.enableRowExpo
        elif self._type == mpz_mpfr:
            return self._core.mpz_mpfr.enableRowExpo
        else:
            raise RuntimeError("MatGSO object '%s' has no core."%self)

    @property
    def transform_enabled(self):
        if self._type == mpz_double:
            return self._core.mpz_double.enableTransform
        elif self._type == mpz_mpfr:
            return self._core.mpz_mpfr.enableTransform
        else:
            raise RuntimeError("MatGSO object '%s' has no core."%self)

    @property
    def inv_transform_enabled(self):
        if self._type == mpz_double:
            return self._core.mpz_double.enableInvTransform
        elif self._type == mpz_mpfr:
            return self._core.mpz_mpfr.enableInvTransform
        else:
            raise RuntimeError("MatGSO object '%s' has no core."%self)

    @property
    def row_op_force_long(self):
        if self._type == mpz_double:
            return self._core.mpz_double.rowOpForceLong
        elif self._type == mpz_mpfr:
            return self._core.mpz_mpfr.rowOpForceLong
        else:
            raise RuntimeError("MatGSO object '%s' has no core."%self)

    def row_op_begin(self, int first, int last):
        if self._type == mpz_double:
            return self._core.mpz_double.rowOpBegin(first, last)
        elif self._type == mpz_mpfr:
            return self._core.mpz_mpfr.rowOpBegin(first, last)
        else:
            raise RuntimeError("MatGSO object '%s' has no core."%self)

    def row_op_end(self, int first, int last):
        if self._type == mpz_double:
            return self._core.mpz_double.rowOpEnd(first, last)
        elif self._type == mpz_mpfr:
            return self._core.mpz_mpfr.rowOpEnd(first, last)
        else:
            raise RuntimeError("MatGSO object '%s' has no core."%self)

    def row_ops(self, int first, int last):
        """FIXME! briefly describe function

        :param int first:
        :param int last:

        """
        return MatGSORowOpContext(self, first, last)

    def get_gram(self, int i, int j):
        preprocess_indices(i, j, self.d, self.d)

        cdef FP_NR[double] t_double
        cdef FP_NR[mpfr_t] t_mpfr

        if self._type == mpz_double:
            self._core.mpz_double.getGram(t_double, i, j)
            r = t_double.getData()
            return r
        elif self._type == mpz_mpfr:
            self._core.mpz_mpfr.getGram(t_mpfr, i, j)
            raise NotImplementedError("converting from MPFR not implemented")
        else:
            raise RuntimeError("MatGSO object '%s' has no core."%self)


    def get_r(self, int i, int j):
        """FIXME! briefly describe function

        :param i:
        :param j:
        :returns:
        :rtype: double

        """
        preprocess_indices(i, j, self.d, self.d)
        cdef FP_NR[double] t_double
        cdef FP_NR[mpfr_t] t_mpfr

        if self._type == mpz_double:
            self._core.mpz_double.getR(t_double, i, j)
            return t_double.getData()
        elif self._type == mpz_mpfr:
            # TODO: don't just return doubles
            self._core.mpz_mpfr.getR(t_mpfr, i, j)
            return t_mpfr.get_d()
        else:
            raise RuntimeError("MatGSO object '%s' has no core."%self)

    def get_r_exp(self, int i, int j):
        """FIXME! briefly describe function

        :param i:
        :param j:
        :returns:
        :rtype: double

        """
        preprocess_indices(i, j, self.d, self.d)
        cdef double r = 0.0
        cdef long expo = 0

        if self._type == mpz_double:
            r = self._core.mpz_double.getRExp(i, j, expo).getData()
        elif self._type == mpz_mpfr:
            # TODO: don't just return doubles
            r = self._core.mpz_mpfr.getRExp(i, j, expo).get_d()
        else:
            raise RuntimeError("MatGSO object '%s' has no core."%self)

        return r, expo


    def update_gso(self):
        if self._type == mpz_double:
            return bool(self._core.mpz_double.updateGSO())
        elif self._type == mpz_mpfr:
            return bool(self._core.mpz_mpfr.updateGSO())
        else:
            raise RuntimeError("MatGSO object '%s' has no core."%self)

    def discover_all_rows(self):
        if self._type == mpz_double:
            self._core.mpz_double.discoverAllRows()
        elif self._type == mpz_mpfr:
            self._core.mpz_mpfr.discoverAllRows()
        else:
            raise RuntimeError("MatGSO object '%s' has no core."%self)

    def move_row(self, int old_r, int new_r):
        preprocess_indices(old_r, new_r, self.d, self.d)
        if self._type == mpz_double:
            return self._core.mpz_double.moveRow(old_r, new_r)
        elif self._type == mpz_mpfr:
            return self._core.mpz_mpfr.moveRow(old_r, new_r)
        else:
            raise RuntimeError("MatGSO object '%s' has no core."%self)


        # void setR(int i, int j, FT& f)
        # void swapRows(int row1, int row2)

    def row_addmul(self, int i, int j, x):
        preprocess_indices(i, j, self.d, self.d)
        cdef double x_ = x
        cdef FP_NR[double] x_double
        cdef FP_NR[mpfr_t] x_mpfr

        if self._type == mpz_double:
            x_double = x_
            self._core.mpz_double.row_addmul(i, j, x_double)
        elif self._type == mpz_mpfr:
            x_mpfr = x_
            self._core.mpz_mpfr.row_addmul(i, j, x_mpfr)
        else:
            raise RuntimeError("MatGSO object '%s' has no core."%self)

    def create_row(self):
        if self._type == mpz_double:
            return self._core.mpz_double.createRow()
        elif self._type == mpz_mpfr:
            return self._core.mpz_mpfr.createRow()
        else:
            raise RuntimeError("MatGSO object '%s' has no core."%self)

    def remove_last_row(self):
        if self._type == mpz_double:
            return self._core.mpz_double.removeLastRow()
        elif self._type == mpz_mpfr:
            return self._core.mpz_mpfr.removeLastRow()
        else:
            raise RuntimeError("MatGSO object '%s' has no core."%self)
