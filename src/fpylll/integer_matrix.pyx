# -*- coding: utf-8 -*-
# distutils: language = c++
# distutils: libraries = gmp mpfr fplll
"""
.. moduleauthor:: Martin R. Albrecht <martinralbrecht+fpylll@googlemail.com>
"""

include "interrupt/interrupt.pxi"

from fplll cimport MatrixRow, sqrNorm, Z_NR
from fpylll.util cimport preprocess_indices, assign_Z_NR_mpz, assign_mpz
from fpylll.gmp.pylong cimport mpz_get_pyintlong

import re
from math import log, ceil, sqrt

from gmp.mpz cimport mpz_init, mpz_mod, mpz_fdiv_q_ui, mpz_clear, mpz_cmp, mpz_sub, mpz_set

cdef class IntegerMatrixRow:
    """
    """
    def __init__(self, IntegerMatrix M, int row):
        """FIXME! briefly describe function

        :param IntegerMatrix M:
        :param int row:
        :returns:
        :rtype:

        """
        preprocess_indices(row, row, M.nrows, M.nrows)
        self.row = row
        self.m = M

    def __getitem__(self, int column):
        """FIXME! briefly describe function

        :param int column:
        :returns:
        :rtype:

        """
        preprocess_indices(column, column, self.m._core.getCols(), self.m._core.getCols())
        r = mpz_get_pyintlong(self.m._core[0][self.row][column].getData())
        return r

    def __str__(self):
        """FIXME! briefly describe function

        :returns:
        :rtype:

        """
        cdef int i
        r = []
        for i in range(self.m._core.getCols()):
            t = mpz_get_pyintlong(self.m._core[0][self.row][i].getData())
            r.append(str(t))
        return "(" + ", ".join(r) + ")"

    def __repr__(self):
        return "row %d of %r"%(self.row, self.m)

    def norm(self):
        """FIXME! briefly describe function

        :returns:
        :rtype:

        """
        cdef Z_NR[mpz_t] t
        sqrNorm[Z_NR[mpz_t]](t, self.m._core[0][self.row], self.m._core.getCols())
        # TODO: don't just use doubles
        return sqrt(t.get_d())

    def __abs__(self):
        """FIXME! briefly describe function

        :returns:
        :rtype:

        """
        return self.norm()


cdef class IntegerMatrix:
    """
    Dense matrices over the Integers.
    """
    def __init__(self, int nrows, int ncols):
        """Construct a new integer matrix

        :param int nrows: number of rows ≥ 0
        :param int ncols: number of columns ≥ 0

        >>> from fpylll import IntegerMatrix
        >>> IntegerMatrix(10, 10) # doctest: +ELLIPSIS
        <IntegerMatrix(10, 10) at 0x...>

        >>> IntegerMatrix(10, 0) # doctest: +ELLIPSIS
        <IntegerMatrix(10, 0) at 0x...>

        >>> IntegerMatrix(-1,  0)
        Traceback (most recent call last):
        ...
        ValueError: Number of rows must be >0

        """
        # TODO: support IntegerMatrix(A)
        # TODO: IntegerMatrix(list)
        # TODO: IntegerMatrix(list of list)

        if nrows < 0:
            raise ValueError("Number of rows must be >0")

        if ncols < 0:
            raise ValueError("Number of columns must be >0")

        self._core = new ZZ_mat[mpz_t](nrows, ncols)

    def __dealloc__(self):
        """
        Delete integer matrix
        """
        del self._core

    def __repr__(self):
        """FIXME! briefly describe function

        :returns:
        :rtype:

        """
        return "<IntegerMatrix(%d, %d) at %s>" % (
            self._core.getRows(),
            self._core.getCols(),
            hex(id(self)))

    def __str__(self):
        """FIXME! briefly describe function

        :returns:
        :rtype:

        """
        cdef int i, j
        max_length = []
        for j in range(self._core.getCols()):
            max_length.append(1)
            for i in range(self._core.getRows()):
                value = self[i, j]
                if not value:
                    continue
                length = ceil(log(abs(value), 10))
                # add one if clean multiple of 10
                length += int(not (abs(value) % 10))
                length += int(value < 0)
                if length > max_length[j]:
                    max_length[j] = int(length)

        r = []
        for i in range(self._core.getRows()):
            r.append(["["])
            for j in range(self._core.getCols()):
                r[-1].append(("%%%dd"%max_length[j])%self[i,j])
            r[-1].append("]")
            r[-1] = " ".join(r[-1])
        r = "\n".join(r)
        return r

    def __copy__(self):
        """FIXME! briefly describe function

        :returns:
        :rtype:

        """
        cdef IntegerMatrix A = IntegerMatrix(self.nrows, self.ncols)
        cdef int i, j
        for i in range(self.nrows):
            for j in range(self.ncols):
                A._core[0][i][j] = self._core[0][i][j]
        return A

    @property
    def nrows(self):
        """Number of Rows

        :returns: number of rows

        >>> from fpylll import IntegerMatrix
        >>> IntegerMatrix(10, 10).nrows
        10

        """
        return self._core.getRows()

    @property
    def ncols(self):
        """Number of Columns

        :returns: number of columns

        >>> from fpylll import IntegerMatrix
        >>> IntegerMatrix(10, 10).ncols
        10

        """
        return self._core.getCols()

    def __getitem__(self, key):
        """Select a row or entry.

        :param key: either an integer for the row or a tuple for row and column.
        :returns: a reference to a row or an integer depending on format of ``key``

        >>> from fpylll import IntegerMatrix
        >>> A = IntegerMatrix(10, 10)
        >>> A.gen_identity(10)
        >>> A[1,0]
        0

        >>> print A[1]
        (0, 1, 0, 0, 0, 0, 0, 0, 0, 0)

        """
        cdef int i = 0
        cdef int j = 0

        if isinstance(key, tuple):
            i, j = key
            preprocess_indices(i, j, self._core.getRows(), self._core.getCols())
            r = mpz_get_pyintlong(self._core[0][i][j].getData())
            return r
        elif isinstance(key, int):
            i = key
            preprocess_indices(i, i, self._core.getRows(), self._core.getRows())
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
            TypeError: 'fpylll.integer_matrix.IntegerMatrixRow' object does not support item assignment

        """
        cdef int i = 0
        cdef int j = 0
        cdef mpz_t tmp

        if isinstance(key, tuple):
            i, j = key
            preprocess_indices(i, j, self._core.getRows(), self._core.getCols())
            assign_Z_NR_mpz(self._core[0][i][j], value)

        elif isinstance(key, int):
            i = key
            preprocess_indices(i, i, self._core.getRows(), self._core.getRows())
            raise NotImplementedError
        else:
            raise ValueError("Parameter '%s' not understood."%key)

    def __mul__(IntegerMatrix A, IntegerMatrix B):
        """Naive matrix × matrix products.

        :param IntegerMatrix A: m × n integer matrix A
        :param IntegerMatrix B: n × k integer matrix B
        :returns: m × k integer matrix C = A × B

        >>> from fpylll import set_random_seed
        >>> set_random_seed(1337)
        >>> A = IntegerMatrix(2, 2)
        >>> A.randomize("uniform", bits=2)
        >>> print A
        [ 2 0 ]
        [ 1 3 ]

        >>> B = IntegerMatrix(2, 2)
        >>> B.randomize("uniform", bits=2)
        >>> print B
        [ 3 2 ]
        [ 3 3 ]

        >>> print A*B
        [  6  4 ]
        [ 12 11 ]

        >>> print B*A
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
        """FIXME! briefly describe function

        :param q:
        :returns:
        :rtype:

        """
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

        cdef IntegerMatrix A = IntegerMatrix(self.nrows, self.ncols)

        cdef int i, j
        for i in range(self.nrows):
            for j in range(self.ncols):
                mpz_set(t1, self._core[0][i][j].getData())
                mpz_set(t2, A._core[0][i][j].getData())

                mpz_mod(t2, t1, q_)
                if mpz_cmp(t2, q2_) > 0:
                    mpz_sub(t2, t2, q_)
                A._core[0][i][j].set(t2)

        mpz_clear(q_)
        mpz_clear(q2_)
        mpz_clear(t1)
        mpz_clear(t2)

        return A

    def __richcmp__(IntegerMatrix self, IntegerMatrix other, int op):
        """Compare two matrices.

        :param IntegerMatrix self:
        :param IntegerMatrix other:
        :param int op:
        :returns:
        :rtype:

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
    def randomize(self, algorithm, **kwds):
        """Randomize this matrix using ``algorithm``.

        :param algorithm: string, see below for choices.

        Available algorithms:

        - ``"intrel"`` -
        - ``"simdioph"`` -
        - ``"uniform"`` -
        - ``"ntrulike"`` -
        - ``"ntrulike2"`` -
        - ``"atjai"`` -

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
            bits = int(kwds["bits"])
            q = int(kwds["q"])
            sig_on()
            self._core.gen_ntrulike(bits, q)
            sig_off()

        elif algorithm == "ntrulike2":
            bits = int(kwds["bits"])
            q = int(kwds["q"])
            sig_on()
            self._core.gen_ntrulike2(bits, q)
            sig_off()

        elif algorithm == "atjai":
            alpha = float(kwds["alpha"])
            sig_on()
            self._core.gen_ajtai(alpha)
            sig_off()

        else:
            raise ValueError("Algorithm '%s' unknown."%algorithm)

    def gen_identity(self, int nrows):
        """Generate identity matrix:

        :param nrows: number of rows

        """
        self._core.gen_identity(nrows)

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
            >>> print A
            [ 1 0 0 0 0   752690  1522220  2972677   890755  2612607 ]
            [ 0 1 0 0 0  1522220  2972677   890755  2612607   752690 ]
            [ 0 0 1 0 0  2972677   890755  2612607   752690  1522220 ]
            [ 0 0 0 1 0   890755  2612607   752690  1522220  2972677 ]
            [ 0 0 0 0 1  2612607   752690  1522220  2972677   890755 ]
            [ 0 0 0 0 0  4194319        0        0        0        0 ]
            [ 0 0 0 0 0        0  4194319        0        0        0 ]
            [ 0 0 0 0 0        0        0  4194319        0        0 ]
            [ 0 0 0 0 0        0        0        0  4194319        0 ]
            [ 0 0 0 0 0        0        0        0        0  4194319 ]

        We can either specify start/stop rows and columns::

            >>> print A.submatrix(0,0,2,8)
            [ 1 0 0 0 0   752690  1522220 2972677 ]
            [ 0 1 0 0 0  1522220  2972677  890755 ]

        Or we can give lists of rows, columns explicitly::

            >>> print A.submatrix([0,1,2],range(3,9))
            [ 0 0   752690  1522220 2972677  890755 ]
            [ 0 0  1522220  2972677  890755 2612607 ]
            [ 0 0  2972677   890755 2612607  752690 ]

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
                    it.next()
                    m += 1
            except StopIteration:
                pass

            it = iter(cols)
            try:
                while True:
                    it.next()
                    n += 1
            except StopIteration:
                pass

            A = IntegerMatrix(m, n)

            i = 0
            for row in iter(rows):
                j = 0
                for col in iter(cols):
                    preprocess_indices(row, col, self._core.getRows(), self._core.getCols())
                    A._core[0][i][j].set(self._core[0][row][col].getData())
                    j += 1
                i += 1
            return A
        else:
            if c < 0:
                c %= self._core.getRows()
            if d < 0:
                d %= self._core.getCols()

            preprocess_indices(a, b, self._core.getRows(), self._core.getCols())
            preprocess_indices(c, d, self._core.getRows()+1, self._core.getCols()+1)

            if c < a:
                raise ValueError("Last row (%d) < first row (%d)"%(c,a))
            if d < b:
                raise ValueError("Last column (%d) < first column (%d)"%(d,b))
            i = 0
            m = c - a
            n = d - b
            A = IntegerMatrix(m, n)
            for row in range(a, c):
                j = 0
                for col in range(b, d):
                    A._core[0][i][j].set(self._core[0][row][col].getData())
                    j += 1
                i += 1
            return A


    @classmethod
    def from_file(cls, filename):
        """FIXME! briefly describe function

        :param cls:
        :param filename:
        :returns:
        :rtype:

        """
        A = IntegerMatrix(0, 0)
        with open(filename, 'r') as fh:
            for i, line in enumerate(fh.readlines()):
                line = re.match("\[+(.*) *\]+", line)
                if line is None:
                    continue
                line = line.groups()[0]
                line = line.strip()
                values = map(int, line.split(" "))
                A._core.setRows(i+1)
                A._core.setCols(len(values))
                for j, v in enumerate(values):
                    A[i, j] = values[j]
        return A
