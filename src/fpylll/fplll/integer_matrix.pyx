# -*- coding: utf-8 -*-
include "fpylll/config.pxi"
include "cysignals/signals.pxi"

"""
Integer matrices.

.. moduleauthor:: Martin R. Albrecht <martinralbrecht+fpylll@googlemail.com>
"""

include "cysignals/signals.pxi"

from cpython cimport PyIndex_Check
from fplll cimport Matrix, MatrixRow, sqr_norm, Z_NR
from fpylll.util cimport preprocess_indices
from fpylll.io cimport assign_Z_NR_mpz, assign_mpz, mpz_get_python

import re
from math import log10, ceil, sqrt, floor

from fpylll.gmp.pylong cimport mpz_get_pyintlong
from fpylll.gmp.mpz cimport mpz_init, mpz_mod, mpz_fdiv_q_ui, mpz_clear, mpz_cmp, mpz_sub, mpz_set


cdef class IntegerMatrixRow:
    """
    A reference to a row in an integer matrix.
    """
    def __init__(self, IntegerMatrix M, int row):
        """Create a row reference.

        :param IntegerMatrix M: Integer matrix
        :param int row: row index

        Row references are immutable::

            >>> from fpylll import IntegerMatrix
            >>> A = IntegerMatrix(2, 3)
            >>> A[0,0] = 1; A[0,1] = 2; A[0,2] = 3
            >>> r = A[0]
            >>> r[0]
            1
            >>> r[0] = 1
            Traceback (most recent call last):
            ...
            TypeError: 'fpylll.fplll.integer_matrix.IntegerMatrixRow' object does not support item assignment

        """
        preprocess_indices(row, row, M.nrows, M.nrows)
        self.row = row
        self.m = M

    def __getitem__(self, int column):
        """Return entry at ``column``

        :param int column: integer offset

        """
        preprocess_indices(column, column, self.m._core.get_cols(), self.m._core.get_cols())
        r = mpz_get_python(self.m._core[0][self.row][column].get_data())
        return r

    def __str__(self):
        """String representation of this row.
        """
        cdef int i
        r = []
        for i in range(self.m._core.get_cols()):
            t = mpz_get_python(self.m._core[0][self.row][i].get_data())
            r.append(str(t))
        return "(" + ", ".join(r) + ")"

    def __repr__(self):
        return "row %d of %r"%(self.row, self.m)

    def __reduce__(self):
        """
        Make sure attempts at pickling raise an error until proper pickling is implemented.
        """
        raise NotImplementedError

    def __abs__(self):
        """Return ℓ_2 norm of this vector.

        >>> A = IntegerMatrix.from_iterable(1, 3, [1,2,3])
        >>> A[0].norm()  # doctest: +ELLIPSIS
        3.74165...
        >>> 1*1 + 2*2 + 3*3
        14
        >>> from math import sqrt
        >>> sqrt(14)  # doctest: +ELLIPSIS
        3.74165...

        """
        cdef Z_NR[mpz_t] t
        sqr_norm[Z_NR[mpz_t]](t, self.m._core[0][self.row], self.m._core.get_cols())
        # TODO: don't just use doubles
        return sqrt(t.get_d())

    norm = __abs__

    def __len__(self):
        """
        >>> A = IntegerMatrix.from_matrix([[1,2],[3,4]], 2, 2)
        >>> len(A[0])
        2

        """
        return self.m._core[0][self.row].size()

    def is_zero(self, int frm=0):
        """Return ``True`` if this vector consists of only zeros starting at index ``frm``

        >>> A = IntegerMatrix.from_matrix([[1,0,0]])
        >>> A[0].is_zero()
        False
        >>> A[0].is_zero(1)
        True

        """
        return bool(self.m._core[0][self.row].is_zero(frm))

    def size_nz(self):
        """Index at which an all zero vector starts.

        >>> A = IntegerMatrix.from_matrix([[0,2,3],[0,2,0],[0,0,0]])
        >>> A[0].size_nz()
        3
        >>> A[1].size_nz()
        2
        >>> A[2].size_nz()
        0

        """

        return self.m._core[0][self.row].size_nz()

    def __iadd__(self, IntegerMatrixRow v):
        """

        >>> A = IntegerMatrix.from_matrix([[0,2],[3,4]])
        >>> A[0] += A[1]
        >>> print(A[0])
        (3, 6)
        >>> v = A[0]
        >>> v += A[1]
        >>> print(A[0])
        (6, 10)

        """
        self.m._core[0][self.row].add(v.m._core[0][v.row])
        return self

    def __isub__(self, IntegerMatrixRow v):
        """

        >>> A = IntegerMatrix.from_matrix([[0,2],[3,4]])
        >>> A[0] -= A[1]
        >>> print(A[0])
        (-3, -2)
        >>> v = A[0]
        >>> v -= A[1]
        >>> print(A[0])
        (-6, -6)

        """
        self.m._core[0][self.row].sub(v.m._core[0][v.row])
        return self

    def addmul(self, IntegerMatrixRow v, x=1, int expo=0):
        """In-place add row vector ``2^expo ⋅ x ⋅ v``

        :param IntegerMatrixRow v: row vector
        :param x: multiplier
        :param int expo: scaling exponent.

        >>> A = IntegerMatrix.from_matrix([[0,2],[3,4]])
        >>> A[0].addmul(A[1])
        >>> print(A[0])
        (3, 6)

        >>> A = IntegerMatrix.from_matrix([[0,2],[3,4]])
        >>> A[0].addmul(A[1],x=0)
        >>> print(A[0])
        (0, 2)

        >>> A = IntegerMatrix.from_matrix([[0,2],[3,4]])
        >>> A[0].addmul(A[1],x=1,expo=2)
        >>> print(A[0])
        (12, 18)

        """
        cdef Z_NR[mpz_t] x_
        cdef Z_NR[mpz_t] tmp
        assign_Z_NR_mpz(x_, x)

        self.m._core[0][self.row].addmul_2exp(v.m._core[0][v.row], x_, expo, tmp)
        return


cdef class IntegerMatrix:
    """
    Dense matrices over the Integers.
    """
    def __init__(self, arg0, arg1=None):
        """Construct a new integer matrix

        :param arg0: number of rows ≥ 0 or matrix
        :param arg1: number of columns ≥ 0 or ``None``

        The default constructor takes the number of rows and columns::

            >>> from fpylll import IntegerMatrix
            >>> IntegerMatrix(10, 10) # doctest: +ELLIPSIS
            <IntegerMatrix(10, 10) at 0x...>

            >>> IntegerMatrix(10, 0) # doctest: +ELLIPSIS
            <IntegerMatrix(10, 0) at 0x...>

            >>> IntegerMatrix(-1,  0)
            Traceback (most recent call last):
            ...
            ValueError: Number of rows must be >0

        The default constructor is also a copy constructor::

            >>> A = IntegerMatrix(2, 2)
            >>> A[0,0] = 1
            >>> B = IntegerMatrix(A)
            >>> B[0,0]
            1
            >>> A[0,0] = 2
            >>> B[0,0]
            1

        """
        cdef int i, j

        if PyIndex_Check(arg0) and PyIndex_Check(arg1):
            if arg0 < 0:
                raise ValueError("Number of rows must be >0")

            if arg1 < 0:
                raise ValueError("Number of columns must be >0")

            self._core = new ZZ_mat[mpz_t](arg0, arg1)
            return

        elif isinstance(arg0, IntegerMatrix) and arg1 is None:
            self._core = new ZZ_mat[mpz_t](arg0.nrows, arg0.ncols)
            for i in range(self.nrows):
                for j in range(self.ncols):
                    self._core[0][i][j] = (<IntegerMatrix>arg0)._core[0][i][j]
            return

        else:
            raise TypeError("Parameters arg0 and arg1 not understood")

    @classmethod
    def from_matrix(cls, A, nrows=None, ncols=None):
        """Construct a new integer matrix from matrix-like object A

        :param A: a matrix like object, with element access A[i,j] or A[i][j]
        :param nrows: number of rows (optional)
        :param ncols: number of columns (optional)


        >>> A = IntegerMatrix.from_matrix([[1,2,3],[4,5,6]])
        >>> print(A)
        [ 1 2 3 ]
        [ 4 5 6 ]

        """
        cdef int m, n

        if nrows is None:
            if hasattr(A, "nrows"):
                nrows = A.nrows
            elif hasattr(A, "__len__"):
                nrows = len(A)
            else:
                raise ValueError("Cannot determine number of rows.")
            if not PyIndex_Check(nrows):
                if callable(nrows):
                    nrows = nrows()
                else:
                    raise ValueError("Cannot determine number of rows.")

        if ncols is None:
            if hasattr(A, "ncols"):
                ncols = A.ncols
            elif hasattr(A[0], "__len__"):
                ncols = len(A[0])
            else:
                raise ValueError("Cannot determine number of rows.")
            if not PyIndex_Check(ncols):
                if callable(ncols):
                    ncols = ncols()
                else:
                    raise ValueError("Cannot determine number of rows.")

        m = nrows
        n = ncols

        B = cls(m, n)
        B.set_matrix(A)
        return B

    @classmethod
    def from_iterable(cls, nrows, ncols, it):
        """Construct a new integer matrix from matrix-like object A

        :param nrows: number of rows
        :param ncols: number of columns
        :param it: an iterable of length at least ``nrows * ncols``

        >>> A = IntegerMatrix.from_iterable(2,3, [1,2,3,4,5,6])
        >>> print(A)
        [ 1 2 3 ]
        [ 4 5 6 ]

        """
        A = cls(nrows, ncols)
        A.set_iterable(it)
        return A

    @classmethod
    def identity(cls, nrows):
        """Construct a new identity matrix of dimension ``nrows × nrows``

        :param nrows: number of rows.

        >>> A = IntegerMatrix.identity(4)
        >>> print(A)
        [ 1 0 0 0 ]
        [ 0 1 0 0 ]
        [ 0 0 1 0 ]
        [ 0 0 0 1 ]

        """
        A = IntegerMatrix(nrows, nrows)
        A.gen_identity(nrows)
        return A

    @classmethod
    def random(cls, d, algorithm, **kwds):
        """Construct new random matrix.

        :seealso: `IntegerMatrix.randomize`
        """
        if algorithm == "intrel":
            A = IntegerMatrix(d, d+1)
        elif algorithm == "simdioph":
            A = IntegerMatrix(d, d)
        elif algorithm == "uniform":
            A = IntegerMatrix(d, d)
        elif algorithm == "ntrulike":
            A = IntegerMatrix(2*d, 2*d)
        elif algorithm == "ntrulike2":
            A = IntegerMatrix(2*d, 2*d)
        elif algorithm == "qary":
            A = IntegerMatrix(d, d)
        elif algorithm == "trg":
            A = IntegerMatrix(d, d)
        else:
            raise ValueError("Algorithm '%s' unknown."%algorithm)
        A.randomize(algorithm, **kwds)
        return A


    def set_matrix(self, A):
        """Set this matrix from matrix-like object A

        :param A: a matrix like object, with element access A[i,j] or A[i][j]

        .. warning:: entries starting at ``A[nrows, ncols]`` are ignored.

        """
        cdef int i, j
        cdef int m = self.nrows
        cdef int n = self.ncols

        try:
            for i in range(m):
                for j in range(n):
                    self[i, j] = A[i, j]
        except TypeError:
            for i in range(m):
                for j in range(n):
                    self[i, j] = A[i][j]


    def set_iterable(self, A):
        """Set this matrix from iterable A

        :param A: an iterable object such as a list or tuple

        .. warning:: entries starting at ``A[nrows * ncols]`` are ignored.

        """
        cdef int i, j
        cdef int m = self.nrows
        cdef int n = self.ncols
        it = iter(A)

        for i in range(m):
            for j in range(n):
                self[i, j] = next(it)


    def to_matrix(self, A):
        """Write this matrix to matrix-like object A

        :param A: a matrix like object, with element access A[i,j] or A[i][j]
        :returns: A

        """
        cdef int i, j
        cdef int m = self.nrows
        cdef int n = self.ncols

        try:
            for i in range(m):
                for j in range(n):
                    A[i, j] = self[i, j]
        except TypeError:
            for i in range(m):
                for j in range(n):
                    A[i][j] = A[i][j]
        return A

    def __dealloc__(self):
        """
        Delete integer matrix
        """
        del self._core

    def __repr__(self):
        """Short representation.

        """
        return "<IntegerMatrix(%d, %d) at %s>" % (
            self._core.get_rows(),
            self._core.get_cols(),
            hex(id(self)))

    def __str__(self):
        """Full string representation of this matrix.

        """
        cdef int i, j
        max_length = []
        for j in range(self._core.get_cols()):
            max_length.append(1)
            for i in range(self._core.get_rows()):
                value = self[i, j]
                if not value:
                    continue
                length = ceil(log10(abs(value)))
                length += int(ceil(log10(abs(value))) == floor(log10(abs(value))))
                # sign
                length += int(value < 0)
                if length > max_length[j]:
                    max_length[j] = int(length)

        r = []
        for i in range(self._core.get_rows()):
            r.append(["["])
            for j in range(self._core.get_cols()):
                r[-1].append(("%%%dd"%max_length[j])%self[i,j])
            r[-1].append("]")
            r[-1] = " ".join(r[-1])
        r = "\n".join(r)
        return r

    def __copy__(self):
        """Copy this matrix.
        """
        cdef IntegerMatrix A = IntegerMatrix(self.nrows, self.ncols)
        cdef int i, j
        for i in range(self.nrows):
            for j in range(self.ncols):
                A._core[0][i][j] = self._core[0][i][j]
        return A

    def __reduce__(self):
        """Serialize this matrix

        >>> import pickle
        >>> A = IntegerMatrix.random(10, "uniform", bits=20)
        >>> pickle.loads(pickle.dumps(A)) == A
        True

        """
        cdef int i, j
        l = []
        for i in range(self._core.get_rows()):
            for j in range(self._core.get_cols()):
                # mpz_get_pyintlong ensure pickles work between Sage & not-Sage
                l.append(int(mpz_get_pyintlong(self._core[0][i][j].get_data())))
        return unpickle_IntegerMatrix, (self.nrows, self.ncols, l)

    @property
    def nrows(self):
        """Number of Rows

        :returns: number of rows

        >>> from fpylll import IntegerMatrix
        >>> IntegerMatrix(10, 10).nrows
        10

        """
        return self._core.get_rows()

    @property
    def ncols(self):
        """Number of Columns

        :returns: number of columns

        >>> from fpylll import IntegerMatrix
        >>> IntegerMatrix(10, 10).ncols
        10

        """
        return self._core.get_cols()

    def __getitem__(self, key):
        """Select a row or entry.

        :param key: an integer for the row, a tuple for row and column or a slice.
        :returns: a reference to a row or an integer depending on format of ``key``

        >>> from fpylll import IntegerMatrix
        >>> A = IntegerMatrix(10, 10)
        >>> A.gen_identity(10)
        >>> A[1,0]
        0

        >>> print(A[1])
        (0, 1, 0, 0, 0, 0, 0, 0, 0, 0)

        >>> print(A[0:2])
        [ 1 0 0 0 0 0 0 0 0 0 ]
        [ 0 1 0 0 0 0 0 0 0 0 ]

        """
        cdef int i = 0
        cdef int j = 0

        if isinstance(key, tuple):
            i, j = key
            preprocess_indices(i, j, self._core.get_rows(), self._core.get_cols())
            r = mpz_get_python(self._core[0][i][j].get_data())
            return r
        elif isinstance(key, slice):
            key = range(*key.indices(self.nrows))
            return self.submatrix(key, range(self.ncols))
        elif PyIndex_Check(key):
            i = key
            preprocess_indices(i, i, self._core.get_rows(), self._core.get_rows())
            return IntegerMatrixRow(self, i)
        else:
            raise ValueError("Parameter '%s' not understood."%key)

    def __setitem__(self, key, value):
        """
        Assign value to index.

        :param key: a tuple of row and column indices
        :param value: an integer

        EXAMPLE::

            >>> from fpylll import IntegerMatrix
            >>> A = IntegerMatrix(10, 10)
            >>> A.gen_identity(10)
            >>> A[1,0] = 2
            >>> A[1,0]
            2

        The notation ``A[i][j]`` is not supported.  This is because ``A[i]`` returns an object
        of type ``IntegerMatrixRow`` object which is immutable by design.  This is to avoid the
        user confusing such an object with a proper vector.::

            >>> A[1][0] = 2
            Traceback (most recent call last):
            ...
            TypeError: 'fpylll.fplll.integer_matrix.IntegerMatrixRow' object does not support item assignment

        """
        cdef int i = 0
        cdef int j = 0

        if isinstance(key, tuple):
            i, j = key
            preprocess_indices(i, j, self._core.get_rows(), self._core.get_cols())
            assign_Z_NR_mpz(self._core[0][i][j], value)

        elif isinstance(key, int):
            i = key
            preprocess_indices(i, i, self._core.get_rows(), self._core.get_rows())
            if isinstance(value, IntegerMatrixRow) and (<IntegerMatrixRow>value).row == i and (<IntegerMatrixRow>value).m == self:
                pass
            else:
                raise NotImplementedError
        else:
            raise ValueError("Parameter '%s' not understood."%key)

    def randomize(self, algorithm, **kwds):
        """Randomize this matrix using ``algorithm``.

        :param algorithm: string, see below for choices.

            Available algorithms:

                - ``"intrel"`` - generate a knapsack like matrix of dimension ``d x (d+1)`` and
                  ``bits`` bits: the i-th vector starts with a random integer of bit-length <=b and
                  the rest is the i-th canonical unit vector.

                - ``"simdioph"`` - generate a ``d x d`` matrix of a form similar to that is involved
                  when trying to find rational approximations to reals with the same small
                  denominator.  The first vector starts with a random integer of bit-length
                  ``<=bits2`` and continues with ``d-1`` independent integers of bit-lengths
                  ``<=bits``; the i-th vector for ``i>1`` is the i-th canonical unit vector scaled
                  by a factor ``2^b``.

                - ``"uniform"`` - generate a ``d x d`` matrix whose entries are independent integers
                  of bit-lengths ``<=bits``.

                - ``"ntrulike"`` - generate an NTRU-like matrix.  If ``bits`` is given, then it
                  first samples an integer ``q`` of bit-length ``<=bits``, whereas if ``q``, then it
                  sets ``q`` to the provided value.  Then it samples a uniform ``h`` in the ring
                  ``Z_q[x]/(x^n-1)``.  It finally returns the 2 x 2 block matrix ``[[I, Rot(h)], [0,
                  q*I]]``, where each block is ``d x d``, the first row of ``Rot(h)`` is the
                  coefficient vector of ``h``, and the i-th row of ``Rot(h)`` is the shift of the
                  (i-1)-th (with last entry put back in first position), for all i>1.  Warning: this
                  does not produce a genuine ntru lattice with h a genuine public key.

                - ``ntrulike2"`` : as the previous option, except that the contructed matrix is
                  ``[[q*I, 0], [Rot(h), I]]``.

                - ``"qary"`` : generate a q-ary matrix.  If ``bits`` is given, then it first samples
                  an integer ``q`` of bit-length ``<=bits``; if ``q`` is provided, then set ``q`` to
                  the provided value.  It returns a ``2 x 2`` block matrix ``[[q*I, 0], [H, I]]``,
                  where ``H`` is ``k x (d-k)`` and uniformly random modulo q.  These bases
                  correspond to the SIS/LWE q-ary lattices.  Goldstein-Mayer lattices correspond to
                  ``k=1`` and ``q`` prime.

                - ``"trg"`` - generate a ``d x d`` lower-triangular matrix ``B`` with ``B_ii =
                  2^(d-i+1)^f`` for all ``i``, and ``B_ij`` is uniform between ``-B_jj/2`` and
                  ``B_jj/2`` for all ``j<i``.
        """
        if algorithm == "intrel":
            bits = int(kwds["bits"])
            sig_on()
            self._core.gen_intrel(bits)
            sig_off()

        elif algorithm == "simdioph":
            bits = int(kwds["bits"])
            bits2 = int(kwds["bits2"])
            self._core.gen_simdioph(bits, bits2)

        elif algorithm == "uniform":
            bits = int(kwds["bits"])
            sig_on()
            self._core.gen_uniform(bits)
            sig_off()

        elif algorithm == "ntrulike":
            if "q" in kwds:
                q = int(kwds["q"])
                sig_on()
                self._core.gen_ntrulike_withq(q)
                sig_off()
            elif "bits" in kwds:
                bits = int(kwds["bits"])
                sig_on()
                self._core.gen_ntrulike(bits)
                sig_off()
            else:
                raise ValueError("Either 'q' or 'bits' is required.")

        elif algorithm == "ntrulike2":
            if "q" in kwds:
                q = int(kwds["q"])
                sig_on()
                self._core.gen_ntrulike2_withq(q)
                sig_off()
            elif "bits" in kwds:
                bits = int(kwds["bits"])
                sig_on()
                self._core.gen_ntrulike2(bits)
                sig_off()
            else:
                raise ValueError("Either 'q' or 'bits' is required.")

        elif algorithm == "qary":
            k = int(kwds["k"])
            if "q" in kwds:
                q = int(kwds["q"])
                sig_on()
                self._core.gen_qary_withq(k, q)
                sig_off()
            elif "bits" in kwds:
                bits = int(kwds["bits"])
                sig_on()
                self._core.gen_qary_prime(k, bits)
                sig_off()
            else:
                raise ValueError("Either 'q' or 'bits' is required.")

        elif algorithm == "trg":
            alpha = float(kwds["alpha"])
            sig_on()
            self._core.gen_trg(alpha)
            sig_off()

        else:
            raise ValueError("Algorithm '%s' unknown."%algorithm)

    def gen_identity(self, int nrows):
        """Generate identity matrix.

        :param nrows: number of rows

        """
        self._core.gen_identity(nrows)


    def clear(self):
        """


        """
        return (<Matrix[Z_NR[mpz_t]]*>self._core).clear()

    def is_empty(self):
        """


        """
        return bool((<Matrix[Z_NR[mpz_t]]*>self._core).empty())

    def resize(self, int rows, int cols):
        """

        :param int rows:
        :param int cols:

        """
        return (<Matrix[Z_NR[mpz_t]]*>self._core).resize(rows, cols)

    def set_rows(self, int rows):
        """

        :param int rows:

        """
        (<Matrix[Z_NR[mpz_t]]*>self._core).set_rows(rows)

    def set_cols(self, int cols):
        """

        :param int cols:

        """
        (<Matrix[Z_NR[mpz_t]]*>self._core).set_cols(cols)

    def swap_rows(self, int r1, int r2):
        """

        :param int r1:
        :param int r2:

        >>> A = IntegerMatrix.from_matrix([[0,2],[3,4]])
        >>> A.swap_rows(0, 1)
        >>> print(A)
        [ 3 4 ]
        [ 0 2 ]


        """
        return (<Matrix[Z_NR[mpz_t]]*>self._core).swap_rows(r1, r2)

    def rotate_left(self, int first, int last):
        """Row permutation.

        ``(M[first],…,M[last])`` becomes ``(M[first+1],…,M[last],M[first])``

        :param int first:
        :param int last:

        >>> A = IntegerMatrix.from_matrix([[0,2],[3,4]])

        """
        return (<Matrix[Z_NR[mpz_t]]*>self._core).rotate_left(first, last)

    def rotate_right(self, int first, int last):
        """Row permutation.

        ``(M[first],…,M[last])`` becomes ``(M[last],M[first],…,M[last-1])``

        :param int first:
        :param int last:

        >>> A = IntegerMatrix.from_matrix([[0,2],[3,4]])

        """
        return (<Matrix[Z_NR[mpz_t]]*>self._core).rotate_right(first, last)

    def rotate(self, int first, int middle, int last):
        """
        Rotates the order of the elements in the range [first,last), in such a way that the element
        pointed by middle becomes the new first element.

        ``(M[first],…,M[middle-1],M[middle],M[last])`` becomes
        ``(M[middle],…,M[last],M[first],…,M[middle-1])``

        :param int first: first index
        :param int middle: new first index
        :param int last: last index (inclusive)

        >>> A = IntegerMatrix.from_matrix([[0,1,2],[3,4,5],[6,7,8]])
        >>> A.rotate(0,0,2)
        >>> print(A)
        [ 0 1 2 ]
        [ 3 4 5 ]
        [ 6 7 8 ]

        >>> A = IntegerMatrix.from_matrix([[0,1,2],[3,4,5],[6,7,8]])
        >>> A.rotate(0,2,2)
        >>> print(A)
        [ 6 7 8 ]
        [ 0 1 2 ]
        [ 3 4 5 ]

        """
        return (<Matrix[Z_NR[mpz_t]]*>self._core).rotate(first, middle, last)

    def rotate_gram_left(self, int first, int last, int n_valid_rows):
        """
        Transformation needed to update the lower triangular Gram matrix when
        ``rotateLeft(first, last)`` is done on the basis of the lattice.

        :param int first:
        :param int last:
        :param int n_valid_rows:

        >>> A = IntegerMatrix.from_matrix([[0,2],[3,4]])

        """
        return (<Matrix[Z_NR[mpz_t]]*>self._core).rotate_gram_left(first, last, n_valid_rows)

    def rotate_gram_right(self, int first, int last, int n_valid_rows):
        """
        Transformation needed to update the lower triangular Gram matrix when
        ``rotateRight(first, last)`` is done on the basis of the lattice.

        :param int first:
        :param int last:
        :param int n_valid_rows:

        >>> A = IntegerMatrix.from_matrix([[0,2],[3,4]])

        """
        return (<Matrix[Z_NR[mpz_t]]*>self._core).rotate_gram_right(first, last, n_valid_rows)

    def transpose(self):
        """
        Transpose.

        >>> A = IntegerMatrix.from_matrix([[0,2],[3,4]])
        >>> A.transpose()
        >>> print(A)
        [ 0 3 ]
        [ 2 4 ]

        """
        return (<Matrix[Z_NR[mpz_t]]*>self._core).transpose()

    def get_max_exp(self):
        """

        >>> A = IntegerMatrix.from_matrix([[0,2],[3,4]])
        >>> A.get_max_exp()
        3

        >>> A = IntegerMatrix.from_matrix([[0,2],[3,9]])
        >>> A.get_max_exp()
        4

        """
        return (<Matrix[Z_NR[mpz_t]]*>self._core).get_max_exp()



# Extensions

    def __mul__(IntegerMatrix A, IntegerMatrix B):
        """Naive matrix × matrix products.

        :param IntegerMatrix A: m × n integer matrix A
        :param IntegerMatrix B: n × k integer matrix B
        :returns: m × k integer matrix C = A × B

        >>> from fpylll import set_random_seed
        >>> set_random_seed(1337)
        >>> A = IntegerMatrix(2, 2)
        >>> A.randomize("uniform", bits=2)
        >>> print(A)
        [ 2 0 ]
        [ 1 3 ]

        >>> B = IntegerMatrix(2, 2)
        >>> B.randomize("uniform", bits=2)
        >>> print(B)
        [ 3 2 ]
        [ 3 3 ]

        >>> print(A*B)
        [  6  4 ]
        [ 12 11 ]

        >>> print(B*A)
        [ 8 6 ]
        [ 9 9 ]

        """
        if A.ncols != B.nrows:
            raise ValueError("Number of columns of A (%d) does not match number of rows of B (%d)"%(A.ncols, B.nrows))

        cdef IntegerMatrix res = IntegerMatrix(A.nrows, B.ncols)
        cdef int i, j
        cdef Z_NR[mpz_t] tmp
        for i in range(A.nrows):
            for j in range(B.ncols):
                tmp = res._core[0][i][j]
                for k in range(A.ncols):
                    tmp.addmul(A._core[0][i][k], B._core[0][k][j])
                res._core[0][i][j] = tmp
        return res

    def __mod__(IntegerMatrix self, q):
        """Return A mod q.

        :param q: a modulus > 0

        """
        A = self.__copy__()
        A.mod(q)
        return A

    def mod(IntegerMatrix self, q, int start_row=0, int start_col=0, int stop_row=-1, int stop_col=-1):
        """Apply moduluar reduction modulo `q` to this matrix.

        :param q: modulus
        :param int start_row: starting row
        :param int start_col: starting column
        :param int stop_row: last row (excluding)
        :param int stop_col: last column (excluding)

        >>> A = IntegerMatrix(2, 2)
        >>> A[0,0] = 1001
        >>> A[1,0] = 13
        >>> A[0,1] = 102
        >>> print(A)
        [ 1001 102 ]
        [   13   0 ]

        >>> A.mod(10, start_row=1, start_col=0)
        >>> print(A)
        [ 1001 102 ]
        [    3   0 ]

        >>> A.mod(10)
        >>> print(A)
        [ 1 2 ]
        [ 3 0 ]

        >>> A = IntegerMatrix(2, 2)
        >>> A[0,0] = 1001
        >>> A[1,0] = 13
        >>> A[0,1] = 102
        >>> A.mod(10, stop_row=1)
        >>> print(A)
        [  1 2 ]
        [ 13 0 ]

        """
        preprocess_indices(start_row, start_col, self.nrows, self.ncols)
        preprocess_indices(stop_row, stop_col, self.nrows+1, self.ncols+1)

        cdef mpz_t q_
        mpz_init(q_)
        try:
            assign_mpz(q_, q)
        except NotImplementedError, msg:
            mpz_clear(q_)
            raise NotImplementedError(msg)

        cdef mpz_t t1
        mpz_init(t1)
        cdef mpz_t t2
        mpz_init(t2)

        cdef mpz_t q2_
        mpz_init(q2_)
        mpz_fdiv_q_ui(q2_, q_, 2)

        cdef int i, j
        for i in range(self.nrows):
            for j in range(self.ncols):
                mpz_set(t1, self._core[0][i][j].get_data())

                if start_row <= i < stop_row and start_col <= i < stop_col:
                    mpz_mod(t2, t1, q_)
                    if mpz_cmp(t2, q2_) > 0:
                        mpz_sub(t2, t2, q_)
                    self._core[0][i][j].set(t2)

        mpz_clear(q_)
        mpz_clear(q2_)
        mpz_clear(t1)
        mpz_clear(t2)

    def __richcmp__(IntegerMatrix self, IntegerMatrix other, int op):
        """Compare two matrices
        """
        cdef int i, j
        cdef Z_NR[mpz_t] a, b
        if op == 2 or op == 3:
            eq = True
            if self.nrows != other.nrows:
                eq = False
            elif self.ncols != other.ncols:
                eq = False
            for i in range(self.nrows):
                if eq is False:
                    break
                for j in range(self.ncols):
                    a = self._core[0][i][j]
                    b = other._core[0][i][j]
                    if a != b:
                        eq = False
                        break
        else:
            raise NotImplementedError("Only != and == are implemented for integer matrices.")
        if op == 2:
            return eq
        elif op == 3:
            return not eq

    def apply_transform(self, IntegerMatrix U, int start_row=0):
        """Apply transformation matrix ``U`` to this matrix starting at row ``start_row``.

        :param IntegerMatrix U: transformation matrix
        :param int start_row: start transformation in this row

        """
        cdef int i, j
        cdef mpz_t tmp
        S = self.submatrix(start_row, 0, start_row + U.nrows, self.ncols)
        cdef IntegerMatrix B = U*S
        for i in range(B.nrows):
            for j in range(B.ncols):
                tmp = B._core[0][i][j].get_data()
                self._core[0][start_row+i][j].set(tmp)


    def submatrix(self, a, b, c=None, d=None):
        """Construct a new submatrix.

        :param a: either the index of the first row or an iterable of row indices
        :param b: either the index of the first column or an iterable of column indices
        :param c: the index of first excluded row (or ``None``)
        :param d: the index of first excluded column (or ``None``)
        :returns:
        :rtype:

        We illustrate the calling conventions of this function using a 10 x 10 matrix::

            >>> from fpylll import IntegerMatrix, set_random_seed
            >>> A = IntegerMatrix(10, 10)
            >>> set_random_seed(1337)
            >>> A.randomize("ntrulike", bits=22, q=4194319)
            >>> print(A)
            [ 1 0 0 0 0 3021421  752690 1522220 2972677  119630 ]
            [ 0 1 0 0 0  119630 3021421  752690 1522220 2972677 ]
            [ 0 0 1 0 0 2972677  119630 3021421  752690 1522220 ]
            [ 0 0 0 1 0 1522220 2972677  119630 3021421  752690 ]
            [ 0 0 0 0 1  752690 1522220 2972677  119630 3021421 ]
            [ 0 0 0 0 0 4194319       0       0       0       0 ]
            [ 0 0 0 0 0       0 4194319       0       0       0 ]
            [ 0 0 0 0 0       0       0 4194319       0       0 ]
            [ 0 0 0 0 0       0       0       0 4194319       0 ]
            [ 0 0 0 0 0       0       0       0       0 4194319 ]

        We can either specify start/stop rows and columns::

            >>> print(A.submatrix(0,0,2,8))
            [ 1 0 0 0 0 3021421  752690 1522220 ]
            [ 0 1 0 0 0  119630 3021421  752690 ]

        Or we can give lists of rows, columns explicitly::

            >>> print(A.submatrix([0,1,2],range(3,9)))
            [ 0 0 3021421  752690 1522220 2972677 ]
            [ 0 0  119630 3021421  752690 1522220 ]
            [ 0 0 2972677  119630 3021421  752690 ]

        """
        cdef int m = 0
        cdef int n = 0
        cdef int i, j, row, col

        if c is None and d is None:
            try:
                iter(a)
                rows = a
                iter(b)
                cols = b
            except TypeError:
                raise ValueError("Inputs to submatrix not understood.")
            it = iter(rows)
            try:
                while True:
                    next(it)
                    m += 1
            except StopIteration:
                pass

            it = iter(cols)
            try:
                while True:
                    next(it)
                    n += 1
            except StopIteration:
                pass

            A = IntegerMatrix(m, n)

            i = 0
            for row in iter(rows):
                j = 0
                for col in iter(cols):
                    preprocess_indices(row, col, self._core.get_rows(), self._core.get_cols())
                    A._core[0][i][j].set(self._core[0][row][col].get_data())
                    j += 1
                i += 1
            return A
        else:
            if c < 0:
                c %= self._core.get_rows()
            if d < 0:
                d %= self._core.get_cols()

            preprocess_indices(a, b, self._core.get_rows(), self._core.get_cols())
            preprocess_indices(c, d, self._core.get_rows()+1, self._core.get_cols()+1)

            if c < a:
                raise ValueError("Last row (%d) < first row (%d)"%(c, a))
            if d < b:
                raise ValueError("Last column (%d) < first column (%d)"%(d, b))
            i = 0
            m = c - a
            n = d - b
            A = IntegerMatrix(m, n)
            for row in range(a, c):
                j = 0
                for col in range(b, d):
                    A._core[0][i][j].set(self._core[0][row][col].get_data())
                    j += 1
                i += 1
            return A

    @classmethod
    def from_file(cls, filename):
        """Construct new matrix from file.

        :param filename: name of file to read from

        """
        A = cls(0, 0)
        with open(filename, 'r') as fh:
            for i, line in enumerate(fh.readlines()):
                line = re.match("\[+([^\]]+) *\]", line)
                if line is None:
                    continue
                line = line.groups()[0]
                line = line.strip()
                line = [e for e in line.split(" ") if e != '']
                values = map(int, line)
                (<IntegerMatrix>A)._core.set_rows(i+1)
                (<IntegerMatrix>A)._core.set_cols(len(values))
                for j, v in enumerate(values):
                    A[i, j] = v
        return A


def unpickle_IntegerMatrix(nrows, ncols, l):
    """Deserialize an integer matrix.

    :param nrows: number of rows
    :param ncols: number of columns
    :param l: list of entries

    """
    return IntegerMatrix.from_iterable(nrows, ncols, l)
