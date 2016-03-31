# -*- coding: utf-8 -*-

#
# General Includes

from gmp.mpz cimport mpz_t
from gmp.random cimport gmp_randstate_t
from libcpp.vector cimport vector
from libcpp.string cimport string

#
# Numbers

cdef extern from "fplll/nr.h" namespace "fplll":

    ctypedef double enumf

    cdef cppclass Z_NR[T]:
        T& getData()
        void set(T d)
        double get_d()
        long exponent()
        void set_str(const char* s)
        int cmp(const Z_NR[T]& m)
        int sgn() const

        void operator=(const Z_NR[T]& z);
        void operator=(const mpz_t& z);
        void operator=(long i);
        int operator<(const Z_NR[T]& a)
        int operator<(long a)
        int operator>(const Z_NR[T]& a)
        int operator>(long a)
        int operator<=(const Z_NR[T]& a)
        int operator<=(long a)
        int operator>=(const Z_NR[T]& a)
        int operator>=(long a)
        int operator==(const Z_NR[T]& a)
        int operator==(long a)
        int operator!=(const Z_NR[T]& a)
        int operator!=(long a)

        void add(const Z_NR[T]& a, const Z_NR[T]& b)
        void add_ui(const Z_NR[T]& a, unsigned int b)
        void sub(const Z_NR[T]& a, const Z_NR[T]& b)
        void sub_ui(const Z_NR[T]& a, unsigned int b)
        void neg(const Z_NR[T]& a)
        void mul(const Z_NR[T]& a, const Z_NR[T]& b)
        void mul_si(const Z_NR[T]& a, long b)
        void mul_ui(const Z_NR[T]& a, unsigned long b)
        void mul_2si(const Z_NR[T]& a, long b)
        void div_2si(const Z_NR[T]& a, long b)
        void addmul(const Z_NR[T]& a, const Z_NR[T]& b)
        void addmul_ui(const Z_NR[T]& a, unsigned long b)
        void addmul_si(const Z_NR[T]& a, long b)
        void submul(const Z_NR[T]& a, const Z_NR[T]& b)
        void submul_ui(const Z_NR[T]& a, unsigned long b)
        void abs(const Z_NR[T]& a)
        void swap(Z_NR[T]& a)
        void randb(int bits)
        void randb_si(int bits)
        void randm(const Z_NR[T]& max)
        void randm_si(const Z_NR[T]& max)


    cdef cppclass FP_NR[T]:
        T& getData()
        void set(T d)
        double get_d()
        inline void operator=(const FP_NR[T]& a)
        inline void operator=(double a)

        @staticmethod
        unsigned int getprec()

        @staticmethod
        unsigned int setprec(unsigned int)

cdef extern from "fplll/nr.h":
    cdef struct dpe_struct:
        pass
    ctypedef dpe_struct *dpe_t


# Random Numbers


cdef extern from "fplll/nr.h" namespace "fplll":

    cdef cppclass RandGen:
        @staticmethod
        void init()

        @staticmethod
        void initWithSeed(unsigned long seed)

        @staticmethod
        void initWithTime()

        @staticmethod
        void initWithTime2()

        @staticmethod
        int getInitialized()

        @staticmethod
        gmp_randstate_t& getGMPState()


# Definitions & Enums

cdef extern from "fplll/defs.h" namespace "fplll":

    cdef enum RedStatus:
        RED_SUCCESS
        RED_GSO_FAILURE
        RED_BABAI_FAILURE
        RED_LLL_FAILURE
        RED_ENUM_FAILURE
        RED_BKZ_FAILURE
        RED_BKZ_TIME_LIMIT
        RED_BKZ_LOOPS_LIMIT
        RED_STATUS_MAX

    cdef enum LLLFlags:
        LLL_VERBOSE
        LLL_EARLY_RED
        LLL_SIEGEL
        LLL_DEFAULT

    cdef enum BKZFlags:
        BKZ_DEFAULT
        BKZ_VERBOSE
        BKZ_NO_LLL
        BKZ_MAX_LOOPS
        BKZ_MAX_TIME
        BKZ_BOUNDED_LLL
        BKZ_AUTO_ABORT
        BKZ_DUMP_GSO
        BKZ_GH_BND

    cdef enum LLLMethod:
        LM_WRAPPER
        LM_PROVED
        LM_HEURISTIC
        LM_FAST

    cdef enum SVPMethod:
        SVPM_FAST
        SVPM_PROVED

    cdef enum SVPFlags:
        SVP_DEFAULT
        SVP_VERBOSE
        SVP_OVERRIDE_BND

    cdef enum CVPFlags:
        CVP_DEFAULT
        CVP_VERBOSE

    cdef enum IntType:
        ZT_MPZ
        ZT_LONG
        ZT_DOUBLE

    cdef enum FloatType:
        FT_DEFAULT
        FT_DOUBLE
        FT_LONG_DOUBLE
        FT_DD
        FT_QD
        FT_DPE
        FT_MPFR

    cdef enum SVPMethod:
        SVPM_FAST
        SVPM_PROVED



    cdef double LLL_DEF_DELTA
    cdef double LLL_DEF_ETA



# Matrices over the Integers

cdef extern from "fplll/matrix.h" namespace "fplll":
    cdef cppclass MatrixRow[T]:
        T& operator[](int i)
        int size()
        int is_zero()
        int is_zero(int frm)
        int sizeNZ()
        void fill(long value)
        void add(const MatrixRow[T] v)
        void add(const MatrixRow[T] v, int n)
        void sub(const MatrixRow[T] v)
        void sub(const MatrixRow[T] v, int n)
        void addmul_2exp(const MatrixRow[T]& v, const T& x, long expo, T& tmp)
        void addmul_2exp(const MatrixRow[T]& v, const T& x, long expo, int n, T& tmp)
        void addmul_si(const MatrixRow[T]& v, long x)
        void addmul_si(const MatrixRow[T]& v, long x, int n)
        void addmul_si_2exp(const MatrixRow[T]& v, long x, long expo, T& tmp)
        void addmul_si_2exp(const MatrixRow[T]& v, long x, long expo, int n, T& tmp)

    void dotProduct[T](T& result, const MatrixRow[T]& v1, const MatrixRow[T]& v2, int n)
    void dotProduct[T](T& result, const MatrixRow[T]& v1, const MatrixRow[T]& v2)

    cdef cppclass Matrix[T]:
        Matrix()
        Matrix(int r, int c)

        int getRows()
        int getCols()

        T& operator()(int i, int j)
        MatrixRow[T] operator[](int i)

        void clear()
        int empty()
        void resize(int rows, int cols)
        void setRows(int rows)
        void setCols(int cols)
        void swap(Matrix[T]& m)

        void swapRows(int r1, int r2)
        void rotateLeft(int first, int last)
        void rotateRight(int first, int last)
        void rotate(int first, int middle, int last)
        void rotateGramLeft(int first, int last, int nValidRows)
        void rotateGramRight(int first, int last, int nValidRows)
        void transpose()
        long getMaxExp()

    cdef cppclass ZZ_mat[T]:

        ZZ_mat()
        ZZ_mat(int r, int c)

        int getRows()
        int getCols()
        void setRows(int rows)
        void setCols(int cols)

        Z_NR[T]& operator()(int i, int j)
        MatrixRow[Z_NR[T]] operator[](int i)

        void gen_identity(int nrows)
        void gen_intrel(int bits)
        void gen_simdioph(int bits, int bits2)
        void gen_uniform(int bits)
        void gen_ntrulike(int bits, int q)
        void gen_ntrulike2(int bits, int q)
        void gen_ajtai(double alpha)



# Gram Schmidt Orthogonalization

cdef extern from "fplll/gso.h" namespace "fplll":

    cdef enum MatGSOFlags:
        GSO_DEFAULT
        GSO_INT_GRAM
        GSO_ROW_EXPO
        GSO_OP_FORCE_LONG

    cdef cppclass MatGSO[ZT, FT]:
        MatGSO(Matrix[ZT] B, Matrix[ZT] U, Matrix[ZT] UinvT, int flags)

        int d
        Matrix[ZT]& b
        vector[long] rowExpo
        void rowOpBegin(int first, int last)
        void rowOpEnd(int first, int last)
        void getGram(FT& f, int i, int j)

        const Matrix[FT]& getMuMatrix()
        const FT& getMuExp(int i, int j, long& expo)
        const FT& getMuExp(int i, int j)
        void getMu(FT& f, int i, int j)
        void DumpMu_d(double* Mu, int beg, int blocksize)
        void DumpR_d(double* R, int beg, int blocksize)

        const Matrix[FT]& getRMatrix()
        const FT& getRExp(int i, int j, long& expo)
        const FT& getRExp(int i, int j)
        void getR(FT& f, int i, int j)

        long getMaxMuExp(int i, int nColumns)

        int updateGSORow(int i, int lastJ)
        int updateGSORow(int i)
        int updateGSO()

        void discoverAllRows()
        void setR(int i, int j, FT& f)
        void moveRow(int oldR, int newR)
        void swapRows(int row1, int row2)

        void row_addmul(int i, int j, const FT& x)
        void row_addmul_we(int i, int j, const FT& x, long expoAdd)

        void lockCols()
        void unlockCols()

        void createRow()
        void createRows(int nNewRows)

        void removeLastRow()
        void removeLastRows(int nRemovedRows)

        void applyTransform(const Matrix[FT]& transform, int srcBase, int targetBase)
        void applyTransform(const Matrix[FT]& transform, int srcBase)

        void dumpMu_d(double* mu, int offset, int blocksize)
        void dumpMu_d(vector[double] mu, int offset, int blocksize)

        void dumpR_d(double* r, int offset, int blocksize)
        void dumpR_d(vector[double] r, int offset, int blocksize)

        const int enableIntGram
        const int enableRowExpo
        const int enableTransform

        const int enableInvTransform
        const int rowOpForceLong



# LLL

cdef extern from "fplll/lll.h" namespace "fplll":

    cdef cppclass LLLReduction[ZT,FT]:
        LLLReduction(MatGSO[ZT, FT]& m, double delta, double eta, int flags)

        int lll()
        int lll(int kappaMin)
        int lll(int kappaMin, int kappaStart)
        int lll(int kappaMin, int kappaStart, int kappaEnd)
        int sizeReduction()
        int sizeReduction(int kappaMin)
        int sizeReduction(int kappaMin, int kappaEnd)

        int status
        int finalKappa
        int lastEarlyRed
        int zeros
        int nSwaps

    int isLLLReduced[ZT, FT](MatGSO[ZT, FT]& m, double delta, double eta)


# LLL Wrapper

cdef extern from "fplll/wrapper.h" namespace "fplll":

    cdef cppclass Wrapper:
        Wrapper(ZZ_mat[mpz_t]& b, ZZ_mat[mpz_t]& u, ZZ_mat[mpz_t]& uInv,
                double delta, double eta, int flags)
        int lll()
        int status



# Evaluator

cdef extern from "fplll/evaluator.h" namespace "fplll":

    cdef cppclass Evaluator[FT]:
        Evaluator()

        void evalSol(const vector[FT]& newSolCoord,
                     const enumf& newPartialDist, enumf& maxDist, long normExp)

        vector[FT] solCoord
        int newSolFlag


    cdef cppclass FastEvaluator[FT]:
        FastEvaluator()

        void evalSol(const vector[FT]& newSolCoord,
                     const enumf& newPartialDist, enumf& maxDist, long normExp)

        vector[FT] solCoord
        int newSolFlag



# Enumeration

cdef extern from "fplll/enumerate.h" namespace "fplll":
    cdef cppclass Enumeration:
        @staticmethod
        void enumerateDouble(MatGSO[Z_NR[double], FP_NR[double]]& gso,
                             FP_NR[double]& fMaxDist, Evaluator[FP_NR[double]]& evaluator,
                             int first, int last,
                             const vector[double]& pruning)

        @staticmethod
        void enumerate[FT](MatGSO[Z_NR[mpz_t], FT]& gso, FT& fMaxDist, long maxDistExpo,
                           FastEvaluator[FT]& evaluator, const vector[FT]& targetCoord,
                           const vector[FT]& subTree, int first, int last,
                           const vector[double]& pruning, int dual);



# Enumeration

cdef extern from "fplll/svpcvp.h" namespace "fplll":

    int shortestVector(ZZ_mat[mpz_t]& b,
                       vector[Z_NR[mpz_t]] &solCoord,
                       SVPMethod method, int flags)

    int shortestVectorPruning(ZZ_mat[mpz_t]& b, vector[Z_NR[mpz_t]]& solCoord,
                              const vector[double]& pruning, Z_NR[mpz_t]& argIntMaxDist,
                              int flags)

    # Experimental. Do not use.
    int closestVector(ZZ_mat[mpz_t] b, vector[Z_NR[mpz_t]] &intTarget,
                      vector[Z_NR[mpz_t]]& solCoord, int flags)



# BKZ

cdef extern from "fplll/bkz.h" namespace "fplll":

    cdef cppclass BKZParam:
         BKZParam()
         BKZParam(int blockSize)
         BKZParam(int blockSize, double delta)
         BKZParam(int blockSize, double delta, int flags, int maxLoops, int maxTime, int linearPruningLevel,
                  double autoAbort_scale, int autoAbort_maxNoDec)
         BKZParam(int blockSize, double delta, int flags, int maxLoops, int maxTime, int linearPruningLevel,
                  double autoAbort_scale, int autoAbort_maxNoDec, double ghFactor)
         int blockSize
         double delta
         int flags
         int maxLoops
         double maxTime

         double autoAbort_scale
         int autoAbort_maxNoDec

         vector[double] pruning

         double ghFactor

         string dumpGSOFilename

         BKZParam *preprocessing

         void enableLinearPruning(int level)

    cdef cppclass BKZAutoAbort[FT]:
        BKZAutoAbort(MatGSO[Z_NR[mpz_t], FT]& m, int numRows)
        BKZAutoAbort(MatGSO[Z_NR[mpz_t], FT]& m, int numRows, int startRow)

        int testAbort()
        int testAbort(double scale)
        int testAbort(double scale, int maxNoDec)

    void computeGaussHeurDist[FT](MatGSO[Z_NR[mpz_t], FT]& m, FT& maxDist,
                                  long maxDistExpo, int kappa, int blockSize, double ghFactor)

    double getCurrentSlope[FT](MatGSO[Z_NR[mpz_t], FT]& m, int startRow, int stopRow)


# Utility

cdef extern from "fplll/util.h" namespace "fplll":
    void vectMatrixProduct(vector[Z_NR[mpz_t]] &result,
                           vector[Z_NR[mpz_t]] &x,
                           const ZZ_mat[mpz_t] &m)

    void sqrNorm[T](T& result, const MatrixRow[T]& v, int n)



# Highlevel Functions

cdef extern from "fplll/fplll.h" namespace "fplll":

    int lllReduction(ZZ_mat[mpz_t] b, double delta, double eta,
                     LLLMethod method, FloatType floatType,
                     int precision, int flags)
    int lllReduction(ZZ_mat[mpz_t] b, ZZ_mat[mpz_t] u,
                     double delta, double eta,
                     LLLMethod method, FloatType floatType,
                     int precision, int flags)

    int bkzReduction(ZZ_mat[mpz_t] *b, ZZ_mat[mpz_t] *u,
                     BKZParam &param, FloatType floatType, int precision)
    int bkzReduction(ZZ_mat[mpz_t] *b, int blockSize, int flags, FloatType floatType, int precision)

    int hkzReduction(ZZ_mat[mpz_t] b)

    const char* getRedStatusStr (int status)

