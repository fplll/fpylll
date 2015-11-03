# -*- coding: utf-8 -*-
# distutils: language = c++
# distutils: libraries = gmp mpfr fplll

from cpython.int cimport PyInt_AS_LONG
from fpylll.gmp.pylong cimport mpz_get_pyintlong, mpz_set_pylong
from fpylll.gmp.mpz cimport mpz_init, mpz_clear, mpz_set_si
from fpylll.util cimport preprocess_indices

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
        return "IntegerMatrix(%d, %d)" % (
            self._core.getRows(),
            self._core.getCols())

    def __copy__(self):
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

        """
        cdef int i = 0
        cdef int j = 0

        if isinstance(key, tuple):
            i, j = key
            if i < 0:
                i %= self._core.getRows()
            if j < 0:
                j %= self._core.getCols()

            if i >= self._core.getRows():
                raise IndexError("First index must be < %d but got %d."%(self._core.getRows(), i))
            if j >= self._core.getCols():
                raise IndexError("Second index must be < %d but got %d."%(self._core.getCols(), j))

            r = mpz_get_pyintlong(self._core[0][i][j].getData())
            return r
        else:
            raise ValueError("Parameter '%s' not understood."%key)

    def __setitem__(self, key, value):
        """FIXME! briefly describe function

        :param key:
        :param value:
        :returns:

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
            self._core[0][i]

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
            self._core.gen_intrel(bits)

        elif algorithm == "simdioph":
            bits = int(kwds["bits"])
            bits2 = int(kwds["bits2"])
            self._core.gen_simdioph(bits, bits2)

        elif algorithm == "uniform":
            bits = int(kwds["bits"])
            self._core.gen_uniform(bits)

        elif algorithm == "ntrulike":
            bits = int(kwds["bits"])
            q = int(kwds["q"])
            self._core.gen_ntrulike(bits, q)

        elif algorithm == "ntrulike2":
            bits = int(kwds["bits"])
            q = int(kwds["q"])
            self._core.gen_ntrulike2(bits, q)

        elif algorithm == "atjai":
            alpha = float(kwds["alpha"])
            self._core.gen_ajtai(alpha)

        else:
            raise ValueError("Algorithm '%s' unknown."%algorithm)

    def gen_identity(self, int nrows):
        """Generate identity matrix:

        :param nrows: number of rows

        """
        self._core.gen_identity(nrows)
