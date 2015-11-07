# -*- coding: utf-8 -*-
# distutils: language = c++
# distutils: libraries = gmp mpfr fplll
"""
.. module:: integer_matrix

.. moduleauthor:: Martin R. Albrecht <martinralbrecht+fpylll@googlemail.com>
"""

include "interrupt/interrupt.pxi"

from cpython.int cimport PyInt_AS_LONG
from fpylll.gmp.pylong cimport mpz_get_pyintlong, mpz_set_pylong
from fpylll.gmp.mpz cimport mpz_init, mpz_clear, mpz_set_si
from fpylll.util cimport preprocess_indices
from fplll cimport MatrixRow, sqrNorm, Z_NR

import re
from math import log, ceil, sqrt

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
    def __init__(self, int nrows, int ncols):
        """Construct a new integer matrix

        :param int nrows: number of rows ≥ 0
        :param int ncols: number of columns ≥ 0

        >>> from fpylll import IntegerMatrix
        >>> IntegerMatrix(10, 10)
        IntegerMatrix(10, 10)

        >>> IntegerMatrix(10,  0)
        IntegerMatrix(10, 0)

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
        """Delete integer matrix

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
        """FIXME! briefly describe function

        :param key:
        :returns:


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
        FIXME! briefly describe function

        :param key:
        :param value:

        >>> from fpylll import IntegerMatrix
        >>> A = IntegerMatrix(10, 10)
        >>> A.gen_identity(10)

        >>> A[1,0] = 2
        >>> A[1,0]
        2

        The notation A[i][j] is not supported. This is because A[i] returns an ``IntegerMatrixRow``
        object which is immutable by design. This is to avoid the user confusing such an object
        with a proper vector.

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

            mpz_init(tmp)
            if isinstance(value, int):
                mpz_set_si(tmp, PyInt_AS_LONG(value))
            elif isinstance(value, long):
                mpz_set_pylong(tmp, value)
            else:
                mpz_clear(tmp)
                msg = "Only Python ints and longs are currently supported, but got type '%s'"%type(value)
                raise NotImplementedError(msg)

            self._core[0][i][j].set(tmp)
            mpz_clear(tmp)
        elif isinstance(key, int):
            i = key
            preprocess_indices(i, i, self._core.getRows(), self._core.getRows())
            raise NotImplementedError
        else:
            raise ValueError("Parameter '%s' not understood."%key)

    def randomize(self, algorithm, **kwds):
        """Randomize this matrix using 'algorithm;

        :param algorithm: string, see below for choices.

        Available algorithms:

        - "intrel"
        - "simdioph"
        - "uniform"
        - "ntrulike"
        - "ntrulike2"
        - "atjai"

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


    @classmethod
    def from_file(cls, filename):
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
