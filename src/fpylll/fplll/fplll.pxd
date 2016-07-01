# -*- coding: utf-8 -*-

#
# General Includes

from fpylll.gmp.mpz cimport mpz_t
from fpylll.gmp.random cimport gmp_randstate_t
from libcpp.vector cimport vector
from libcpp.string cimport string

#
# Numbers

cdef extern from "fplll/nr/nr.h" namespace "fplll":

    ctypedef double enumf

    cdef cppclass Z_NR[T]:
        T& getData() nogil
        void set(T d) nogil
        double get_d() nogil
        long exponent() nogil
        void set_str(const char* s) nogil
        int cmp(const Z_NR[T]& m) nogil
        int sgn() nogil

        void operator=(const Z_NR[T]& z) nogil
        void operator=(const mpz_t& z) nogil
        void operator=(long i) nogil
        int operator<(const Z_NR[T]& a) nogil
        int operator<(long a) nogil
        int operator>(const Z_NR[T]& a) nogil
        int operator>(long a) nogil
        int operator<=(const Z_NR[T]& a) nogil
        int operator<=(long a) nogil
        int operator>=(const Z_NR[T]& a) nogil
        int operator>=(long a) nogil
        int operator==(const Z_NR[T]& a) nogil
        int operator==(long a) nogil
        int operator!=(const Z_NR[T]& a) nogil
        int operator!=(long a) nogil

        void add(const Z_NR[T]& a, const Z_NR[T]& b) nogil
        void add_ui(const Z_NR[T]& a, unsigned int b) nogil
        void sub(const Z_NR[T]& a, const Z_NR[T]& b) nogil
        void sub_ui(const Z_NR[T]& a, unsigned int b) nogil
        void neg(const Z_NR[T]& a) nogil
        void mul(const Z_NR[T]& a, const Z_NR[T]& b) nogil
        void mul_si(const Z_NR[T]& a, long b) nogil
        void mul_ui(const Z_NR[T]& a, unsigned long b) nogil
        void mul_2si(const Z_NR[T]& a, long b) nogil
        void div_2si(const Z_NR[T]& a, long b) nogil
        void addmul(const Z_NR[T]& a, const Z_NR[T]& b) nogil
        void addmul_ui(const Z_NR[T]& a, unsigned long b) nogil
        void addmul_si(const Z_NR[T]& a, long b) nogil
        void submul(const Z_NR[T]& a, const Z_NR[T]& b) nogil
        void submul_ui(const Z_NR[T]& a, unsigned long b) nogil
        void abs(const Z_NR[T]& a) nogil
        void swap(Z_NR[T]& a) nogil
        void randb(int bits) nogil
        void randb_si(int bits) nogil
        void randm(const Z_NR[T]& max) nogil
        void randm_si(const Z_NR[T]& max) nogil


    cdef cppclass FP_NR[T]:
        T& getData() nogil
        void set(T d) nogil
        double get_d() nogil
        inline void operator=(const FP_NR[T]& a) nogil
        inline void operator=(double a) nogil

        @staticmethod
        unsigned int getprec() nogil

        @staticmethod
        unsigned int setprec(unsigned int) nogil

cdef extern from "fplll/nr/nr.h":
    cdef struct dpe_struct:
        pass
    ctypedef dpe_struct *dpe_t


# Random Numbers


cdef extern from "fplll/nr/nr.h" namespace "fplll":

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

cdef extern from "fplll/nr/matrix.h" namespace "fplll":
    cdef cppclass MatrixRow[T]:
        T& operator[](int i) nogil
        int size() nogil
        int is_zero() nogil
        int is_zero(int frm) nogil
        int sizeNZ() nogil
        void fill(long value) nogil
        void add(const MatrixRow[T] v) nogil
        void add(const MatrixRow[T] v, int n) nogil
        void sub(const MatrixRow[T] v) nogil
        void sub(const MatrixRow[T] v, int n) nogil
        void addmul_2exp(const MatrixRow[T]& v, const T& x, long expo, T& tmp) nogil
        void addmul_2exp(const MatrixRow[T]& v, const T& x, long expo, int n, T& tmp) nogil
        void addmul_si(const MatrixRow[T]& v, long x) nogil
        void addmul_si(const MatrixRow[T]& v, long x, int n) nogil
        void addmul_si_2exp(const MatrixRow[T]& v, long x, long expo, T& tmp) nogil
        void addmul_si_2exp(const MatrixRow[T]& v, long x, long expo, int n, T& tmp) nogil

    void dotProduct[T](T& result, const MatrixRow[T]& v1, const MatrixRow[T]& v2, int n) nogil
    void dotProduct[T](T& result, const MatrixRow[T]& v1, const MatrixRow[T]& v2) nogil

    cdef cppclass Matrix[T]:
        Matrix()
        Matrix(int r, int c)

        int getRows()
        int getCols()

        T& operator()(int i, int j)
        MatrixRow[T] operator[](int i)

        void clear()
        int empty()
        void resize(int rows, int cols) nogil
        void setRows(int rows) nogil
        void setCols(int cols) nogil
        void swap(Matrix[T]& m) nogil

        void swapRows(int r1, int r2) nogil
        void rotateLeft(int first, int last) nogil
        void rotateRight(int first, int last) nogil
        void rotate(int first, int middle, int last) nogil
        void rotateGramLeft(int first, int last, int nValidRows) nogil
        void rotateGramRight(int first, int last, int nValidRows) nogil
        void transpose() nogil
        long getMaxExp() nogil

    cdef cppclass ZZ_mat[T]:

        ZZ_mat()
        ZZ_mat(int r, int c)

        int getRows() nogil
        int getCols() nogil
        void setRows(int rows) nogil
        void setCols(int cols) nogil

        Z_NR[T]& operator()(int i, int j) nogil
        MatrixRow[Z_NR[T]] operator[](int i) nogil

        void gen_identity(int nrows) nogil
        void gen_intrel(int bits) nogil
        void gen_simdioph(int bits, int bits2) nogil
        void gen_uniform(int bits) nogil
        void gen_ntrulike(int bits) nogil
        void gen_ntrulike_withq(int q) nogil
        void gen_ntrulike2(int bits) nogil
        void gen_ntrulike2_withq(int q) nogil
        void gen_qary_withq(int k, int q) nogil
        void gen_qary_prime(int k, int bits) nogil
        void gen_trg(double alpha) nogil



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
        FT& getGram(FT& f, int i, int j)

        const Matrix[FT]& getMuMatrix() nogil
        const FT& getMuExp(int i, int j, long& expo) nogil
        const FT& getMuExp(int i, int j) nogil
        FT& getMu(FT& f, int i, int j) nogil

        const Matrix[FT]& getRMatrix() nogil
        const FT& getRExp(int i, int j, long& expo) nogil
        const FT& getRExp(int i, int j) nogil
        FT& getR(FT& f, int i, int j) nogil

        long getMaxMuExp(int i, int nColumns) nogil

        int updateGSORow(int i, int lastJ) nogil
        int updateGSORow(int i) nogil
        int updateGSO() nogil

        void discoverAllRows() nogil
        void setR(int i, int j, FT& f) nogil
        void moveRow(int oldR, int newR) nogil
        void swapRows(int row1, int row2)

        void row_addmul(int i, int j, const FT& x) nogil
        void row_addmul_we(int i, int j, const FT& x, long expoAdd) nogil

        void lockCols() nogil
        void unlockCols() nogil

        void createRow() nogil
        void createRows(int nNewRows) nogil

        void removeLastRow() nogil
        void removeLastRows(int nRemovedRows) nogil

        void applyTransform(const Matrix[FT]& transform, int srcBase, int targetBase) nogil
        void applyTransform(const Matrix[FT]& transform, int srcBase) nogil

        void dumpMu_d(double* mu, int offset, int blocksize) nogil
        void dumpMu_d(vector[double] mu, int offset, int blocksize) nogil

        void dumpR_d(double* r, int offset, int blocksize) nogil
        void dumpR_d(vector[double] r, int offset, int blocksize) nogil

        const int enableIntGram
        const int enableRowExpo
        const int enableTransform

        const int enableInvTransform
        const int rowOpForceLong



# LLL

cdef extern from "fplll/lll.h" namespace "fplll":

    cdef cppclass LLLReduction[ZT,FT]:
        LLLReduction(MatGSO[ZT, FT]& m, double delta, double eta, int flags)

        int lll() nogil
        int lll(int kappaMin) nogil
        int lll(int kappaMin, int kappaStart) nogil
        int lll(int kappaMin, int kappaStart, int kappaEnd) nogil
        int sizeReduction() nogil
        int sizeReduction(int kappaMin) nogil
        int sizeReduction(int kappaMin, int kappaEnd) nogil

        int status
        int finalKappa
        int lastEarlyRed
        int zeros
        int nSwaps

    int isLLLReduced[ZT, FT](MatGSO[ZT, FT]& m, double delta, double eta) nogil


# LLL Wrapper

cdef extern from "fplll/wrapper.h" namespace "fplll":

    cdef cppclass Wrapper:
        Wrapper(ZZ_mat[mpz_t]& b, ZZ_mat[mpz_t]& u, ZZ_mat[mpz_t]& uInv,
                double delta, double eta, int flags)
        int lll() nogil
        int status



# Evaluator

cdef extern from "fplll/enum/evaluator.h" namespace "fplll":

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

cdef extern from "fplll/enum/enumerate.h" namespace "fplll":
    cdef cppclass Enumeration[FT]:
        Enumeration(MatGSO[Z_NR[mpz_t], FT]& gso, FastEvaluator[FT]& evaluator)

        void enumerate(int first, int last, FT& fMaxDist, long maxDistExpo,
                       const vector[FT]& targetCoord,
                       const vector[double]& subTree,
                       const vector[double]& pruning,
                       int dual)

        long getNodes()



# SVP

cdef extern from "fplll/svpcvp.h" namespace "fplll":
    int shortestVector(ZZ_mat[mpz_t]& b,
                       vector[Z_NR[mpz_t]] &solCoord,
                       SVPMethod method, int flags) nogil

    int shortestVectorPruning(ZZ_mat[mpz_t]& b, vector[Z_NR[mpz_t]]& solCoord,
                              const vector[double]& pruning, int flags) nogil

    # Experimental. Do not use.
    int closestVector(ZZ_mat[mpz_t] b, vector[Z_NR[mpz_t]] &intTarget,
                      vector[Z_NR[mpz_t]]& solCoord, int flags) nogil



# BKZ

cdef extern from "fplll/bkz_params.h" namespace "fplll":

    cdef cppclass Pruning:
        double radius_factor
        vector[double] coefficients
        double probability

        Pruning()

        @staticmethod
        Pruning LinearPruning(int block_size, int level)

    cdef cppclass Strategy:
        vector[Pruning] pruning_parameters
        vector[int] preprocessing_blocksizes

        @staticmethod
        Strategy EmptyStrategy()

        Pruning get_pruning(double radius, double gh)

    cdef cppclass BKZParam:
         BKZParam() nogil
         BKZParam(int block_size) nogil
         BKZParam(int block_size, vector[Strategy] strategies, double delta) nogil
         BKZParam(int block_size, vector[Strategy] strategies, double delta, int flags, int max_loops, int max_time,
                  double auto_abort_scale, int auto_abort_max_no_dec) nogil
         BKZParam(int block_size, vector[Strategy] strategies, double delta, int flags, int max_loops, int max_time,
                  double auto_abort_scale, int auto_abort_max_no_dec, double gh_factor) nogil
         int block_size
         double delta
         int flags
         int max_loops
         double max_time

         double auto_abort_scale
         int auto_abort_max_no_dec

         vector[Strategy] strategies

         double gh_factor

         string dump_gso_filename

    vector[Strategy] load_strategies_json(const char *filename)

cdef extern from "fplll/bkz.h" namespace "fplll":

    cdef cppclass BKZAutoAbort[FT]:
        BKZAutoAbort(MatGSO[Z_NR[mpz_t], FT]& m, int num_rows) nogil
        BKZAutoAbort(MatGSO[Z_NR[mpz_t], FT]& m, int num_rows, int start_row) nogil

        int test_abort() nogil
        int test_abort(double scale) nogil
        int test_abort(double scale, int max_no_dec) nogil

    void compute_gaussian_heuristic[FT](FT& max_dist, long max_dist_expo,
                                        int block_size, FT& root_det_mpfr, double gh_factor) nogil

    double get_current_slope[FT](MatGSO[Z_NR[mpz_t], FT]& m, int startRow, int stopRow) nogil


# Utility

cdef extern from "fplll/util.h" namespace "fplll":
    void vectMatrixProduct(vector[Z_NR[mpz_t]] &result,
                           vector[Z_NR[mpz_t]] &x,
                           const ZZ_mat[mpz_t] &m) nogil

    void sqrNorm[T](T& result, const MatrixRow[T]& v, int n) nogil



# Highlevel Functions

cdef extern from "fplll/fplll.h" namespace "fplll":

    int lllReduction(ZZ_mat[mpz_t] b, double delta, double eta,
                     LLLMethod method, FloatType floatType,
                     int precision, int flags) nogil
    int lllReduction(ZZ_mat[mpz_t] b, ZZ_mat[mpz_t] u,
                     double delta, double eta,
                     LLLMethod method, FloatType floatType,
                     int precision, int flags) nogil

    int bkzReduction(ZZ_mat[mpz_t] *b, ZZ_mat[mpz_t] *u,
                     BKZParam &param, FloatType floatType, int precision) nogil
    int bkzReduction(ZZ_mat[mpz_t] *b, int blockSize, int flags, FloatType floatType, int precision) nogil

    int hkzReduction(ZZ_mat[mpz_t] b) nogil

    const char* getRedStatusStr (int status) nogil
