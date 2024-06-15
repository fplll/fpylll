# -*- coding: utf-8 -*-

#
# General Includes

from fpylll.gmp.mpz cimport mpz_t
from fpylll.mpfr.mpfr cimport mpfr_t
from fpylll.gmp.random cimport gmp_randstate_t
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.pair cimport pair
from libcpp cimport bool
from libcpp.functional cimport function


cdef extern from "<map>" namespace "std":
    cdef cppclass multimap[T, U]:
        cppclass iterator:
            pair[T,U]& operator*()
            iterator operator++()
            iterator operator--()
            bint operator==(iterator)
            bint operator!=(iterator)
            iterator operator=()

        cppclass reverse_iterator:
            pair[T,U]& operator*()
            iterator operator++()
            iterator operator--()
            bint operator==(reverse_iterator)
            bint operator!=(reverse_iterator)
            iterator operator=()

        map()
        U& operator[](T&)
        U& at(T&)
        iterator begin()
        reverse_iterator rbegin()
        iterator end()
        reverse_iterator rend()
        size_t count(T&)
        bint empty()
        void erase(iterator)
        void erase(iterator, iterator)
        size_t erase(T&)
        iterator find(T&)
        pair[iterator, bint] insert(pair[T,U])
        size_t size()


cdef extern from "fplll/fplll_config.h":
    """
    #ifdef FPLLL_WITH_RECURSIVE_ENUM
    #define FPLLL_HAVE_RECURSIVE_ENUM 1
    #else
    #define FPLLL_HAVE_RECURSIVE_ENUM 0
    #endif
    """
    int FPLLL_MAJOR_VERSION
    int FPLLL_MINOR_VERSION
    int FPLLL_MICRO_VERSION

    int FPLLL_MAX_ENUM_DIM
    bool FPLLL_HAVE_RECURSIVE_ENUM
    int FPLLL_MAX_PARALLEL_ENUM_DIM

#
# Numbers

cdef extern from "fplll/nr/nr.h" namespace "fplll":

    ctypedef double enumf

    cdef cppclass Z_NR[T]:
        T& get_data() nogil
        void set "operator=" (T d) nogil
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
        T& get_data() nogil
        double get_d() nogil
        inline void operator=(const FP_NR[T]& a) nogil
        inline void operator=(double a) nogil
        inline void operator=(const char *s) nogil

        @staticmethod
        unsigned int get_prec() nogil

        @staticmethod
        unsigned int set_prec(unsigned int) nogil

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
        void init_with_seed(unsigned long seed)

        @staticmethod
        void init_with_time()

        @staticmethod
        void init_with_time2()

        @staticmethod
        int get_initialized()

        @staticmethod
        gmp_randstate_t& get_gmp_state()


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
        BKZ_SD_VARIANT
        BKZ_SLD_RED

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

    cdef enum CVPMethod:
        CVPM_FAST
        CVPM_PROVED

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

    cdef enum EvaluatorMode:
        EVALMODE_SV
        EVALMODE_CV
        EVALMODE_COUNT
        EVALMODE_PRINT

    cdef double LLL_DEF_DELTA
    cdef double LLL_DEF_ETA


    const double BKZ_DEF_AUTO_ABORT_SCALE
    const int BKZ_DEF_AUTO_ABORT_MAX_NO_DEC
    const double BKZ_DEF_GH_FACTOR
    const double BKZ_DEF_MIN_SUCCESS_PROBABILITY
    const int BKZ_DEF_RERANDOMIZATION_DENSITY

# Vectors (only used in some places)

cdef extern from "fplll/nr/numvect.h" namespace "fplll":

    cdef cppclass NumVect[T]:

        cppclass iterator:
            iterator operator++()
            iterator operator--()
            bint operator==(iterator)
            bint operator!=(iterator)
            iterator operator=()

        NumVect()
        NumVect(const NumVect[T]& v)
        NumVect(int size)
        NumVect(int size, T &t)

        void operator=(NumVect &v)
        void swap(NumVect &v)

        const iterator begin()
        iterator end()
        int size()
        bool empty()
        void resize(int size)
        void resize(int size, const T &t)
        void gen_zero(int size)

        void push_back(const T &t)
        void pop_back()
        T &front()
        T &back()
        void extend(int maxSize)
        void clear()
        T &operator[](int i)

        void add(const NumVect[T] &v, int n)
        void add(const NumVect[T] &v)
        void sub(const NumVect[T] &v, int n)
        void sub(const NumVect[T] &v)
        void mul(const NumVect[T] &v, int n, T c)
        void mul(const NumVect[T] &v, T c)
        void addmul(const NumVect[T] &v, T x, int n)
        void addmul(const NumVect[T] &v, T x)
        void addmul_2exp(const NumVect[T] &v, const T &x, long expo, T &tmp)
        void addmul_2exp(const NumVect[T] &v, const T &x, long expo, int n, T &tmp)
        void addmul_si(const NumVect[T] &v, long x)
        void addmul_si(const NumVect[T] &v, long x, int n)
        void addmul_si_2exp(const NumVect[T] &v, long x, long expo, T &tmp)
        void addmul_si_2exp(const NumVect[T] &v, long x, long expo, int n, T &tmp)

        # (v[first],...,v[last]) becomes (v[first+1],...,v[last],v[first]) */
        void rotate_left(int first, int last)

        # (v[first],...,v[last]) becomes (v[last],v[first],...,v[last-1]) */
        void rotate_right(int first, int last)

        # Returns expo >= 0 such that all elements are < 2^expo.
        long get_max_exponent()

        void fill(long value)

        bool is_zero(int fromCol = 0) const

        int size_nz() const


# Matrices over the Integers

cdef extern from "fplll/nr/matrix.h" namespace "fplll":
    cdef cppclass MatrixRow[T]:
        T& operator[](int i) nogil
        int size() nogil
        int is_zero() nogil
        int is_zero(int frm) nogil
        int size_nz() nogil
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

        void dot_product(T &result, const MatrixRow[T] &v0) nogil
        void dot_product(T &result, const MatrixRow[T] &v0, int n) nogil

    void dot_product[T](T& result, const MatrixRow[T]& v1, const MatrixRow[T]& v2, int n) nogil
    void dot_product[T](T& result, const MatrixRow[T]& v1, const MatrixRow[T]& v2) nogil

    cdef cppclass Matrix[T]:
        Matrix()
        Matrix(int r, int c)

        int get_rows()
        int get_cols()

        T& operator()(int i, int j)
        MatrixRow[T] operator[](int i)

        void clear()
        int empty()
        void resize(int rows, int cols) nogil
        void set_rows(int rows) nogil
        void set_cols(int cols) nogil
        void swap(Matrix[T]& m) nogil

        void swap_rows(int r1, int r2) nogil
        void rotate_left(int first, int last) nogil
        void rotate_right(int first, int last) nogil
        void rotate(int first, int middle, int last) nogil
        void rotate_gram_left(int first, int last, int nValidRows) nogil
        void rotate_gram_right(int first, int last, int nValidRows) nogil
        void transpose() nogil
        long get_max_exp() nogil

    cdef cppclass ZZ_mat[T]:

        ZZ_mat()
        ZZ_mat(int r, int c)

        int get_rows() nogil
        int get_cols() nogil
        void set_rows(int rows) nogil
        void set_cols(int cols) nogil

        Z_NR[T]& operator()(int i, int j) nogil
        MatrixRow[Z_NR[T]] operator[](int i) nogil

        void gen_identity(int nrows) nogil
        void gen_intrel(int bits) nogil
        void gen_simdioph(int bits, int bits2) nogil
        void gen_uniform(int bits) nogil
        void gen_ntrulike(const Z_NR[T] &q) nogil
        void gen_ntrulike_bits(int bits) nogil
        void gen_ntrulike2(const Z_NR[T] &q) nogil
        void gen_ntrulike2_bits(int bits) nogil
        void gen_qary(int k, const Z_NR[T] &q) nogil
        void gen_qary_prime(int k, int bits) nogil
        void gen_trg(double alpha) nogil



# Gram Schmidt Orthogonalization

cdef extern from "fplll/gso.h" namespace "fplll":

    cdef enum MatGSOInterfaceFlags:
        GSO_DEFAULT
        GSO_INT_GRAM
        GSO_ROW_EXPO
        GSO_OP_FORCE_LONG

    cdef cppclass MatGSO[ZT, FT]:
        MatGSO(Matrix[ZT] B, Matrix[ZT] U, Matrix[ZT] UinvT, int flags)

        int d
        Matrix[ZT]& b
        vector[long] row_expo
        void row_op_begin(int first, int last)
        void row_op_end(int first, int last)
        FT& get_gram(FT& f, int i, int j)

        const Matrix[FT]& get_mu_matrix() nogil
        const FT& get_mu_exp(int i, int j, long& expo) nogil
        const FT& get_mu_exp(int i, int j) nogil
        FT& get_mu(FT& f, int i, int j) nogil

        const Matrix[FT]& get_r_matrix() nogil
        const FT& get_r_exp(int i, int j, long& expo) nogil
        const FT& get_r_exp(int i, int j) nogil
        FT& get_r(FT& f, int i, int j) nogil

        long get_max_mu_exp(int i, int nColumns) nogil

        int update_gso_row(int i, int lastJ) nogil
        int update_gso_row(int i) nogil
        int update_gso() nogil

        void discover_all_rows() nogil
        void set_r(int i, int j, FT& f) nogil
        void move_row(int oldR, int newR) nogil
        void row_swap(int row1, int row2) nogil

        void row_addmul(int i, int j, const FT& x) nogil
        void row_addmul_we(int i, int j, const FT& x, long expoAdd) nogil

        void lock_cols() nogil
        void unlock_cols() nogil

        void create_row() nogil
        void create_rows(int nNewRows) nogil

        void remove_last_row() nogil
        void remove_last_rows(int nRemovedRows) nogil

        void apply_transform(const Matrix[FT]& transform, int srcBase, int targetBase) nogil
        void apply_transform(const Matrix[FT]& transform, int srcBase) nogil

        void dump_mu_d(double* mu, int offset, int block_size) nogil
        void dump_mu_d(vector[double] mu, int offset, int block_size) nogil

        void dump_r_d(double* r, int offset, int block_size) nogil
        void dump_r_d(vector[double] r, int offset, int block_size) nogil

        double get_current_slope(int start_row, int stop_row) nogil
        FT get_root_det(int start_row, int stop_row) nogil
        FT get_log_det(int start_row, int stop_row) nogil
        FT get_slide_potential(int start_row, int stop_row, int block_size) nogil

        void to_canonical(vector[FT] &w, const vector[FT] &v, long start) nogil
        void from_canonical(vector[FT] &v, const vector[FT] &w, long start, long dimension) nogil
        int babai(vector[ZT] w, vector[FT] v, int start, int dimension, bool gsa) nogil

        const int enable_int_gram
        const int enable_row_expo
        const int enable_transform

        const int enable_inverse_transform
        const int row_op_force_long

cdef extern from "fplll/gso_gram.h" namespace "fplll":

    cdef cppclass MatGSOGram[ZT, FT]:
        MatGSOGram(Matrix[ZT] B, Matrix[ZT] U, Matrix[ZT] UinvT, int flags)

        long get_max_exp_of_b() nogil
        bool b_row_is_zero(int i) nogil
        int get_cols_of_b() nogil
        int get_rows_of_b() nogil
        void negate_row_of_b(int i) nogil

        void set_g(Matrix[ZT] arg_g)
        void create_rows(int n_new_rows) nogil
        void remove_last_rows(int n_removed_rows) nogil

        void move_row(int old_r, int new_r) nogil

        void row_addmul_we(int i, int j, const FT &x, long expo_add) nogil

        void row_add(int i, int j) nogil
        void row_sub(int i, int j) nogil

        FT &get_gram(FT &f, int i, int j) nogil


cdef extern from "fplll/gso_interface.h" namespace "fplll":

    cdef cppclass MatGSOInterface[ZT, FT]:
        MatGSOInterface(Matrix[ZT] B, Matrix[ZT] U, Matrix[ZT] UinvT, int flags)

        int d

        long get_max_exp_of_b() nogil
        bool b_row_is_zero(int i) nogil

        int get_cols_of_b() nogil
        int get_rows_of_b() nogil

        void negate_row_of_b(int i) nogil
        vector[long] row_expo

        inline void row_op_begin(int first, int last) nogil
        void row_op_end(int first, int last) nogil
        FT &get_gram(FT &f, int i, int j) nogil
        ZT &get_int_gram(ZT &f, int i, int j) nogil
        const Matrix[FT] &get_mu_matrix() nogil
        const Matrix[FT] &get_r_matrix() nogil
        const Matrix[ZT] &get_g_matrix() nogil
        inline const FT &get_mu_exp(int i, int j, long &expo) nogil
        inline const FT &get_mu_exp(int i, int j) nogil
        inline FT &get_mu(FT &f, int i, int j) nogil
        ZT get_max_gram() nogil
        FT get_max_bstar() nogil
        inline const FT &get_r_exp(int i, int j, long &expo) nogil
        inline const FT &get_r_exp(int i, int j) nogil
        inline FT &get_r(FT &f, int i, int j) nogil
        long get_max_mu_exp(int i, int n_columns) nogil
        bool update_gso_row(int i, int last_j) nogil

        inline bool update_gso_row(int i) nogil
        inline bool update_gso() nogil

        inline void discover_all_rows() nogil
        void set_r(int i, int j, FT &f) nogil

        void move_row(int old_r, int new_r) nogil
        void row_swap(int row1, int row2) nogil


        inline void row_addmul(int i, int j, const FT &x) nogil
        void row_addmul_we(int i, int j, const FT &x, long expo_add) nogil
        void row_add(int i, int j) nogil
        void row_sub(int i, int j) nogil
        void lock_cols() nogil
        void unlock_cols() nogil
        inline void create_row() nogil
        void create_rows(int n_new_rows) nogil
        inline void remove_last_row() nogil
        void remove_last_rows(int n_removed_rows) nogil

        void apply_transform(const Matrix[FT] &transform, int src_base, int target_base) nogil
        void apply_transform(const Matrix[FT] &transform, int src_base) nogil

        void dump_mu_d(double* mu, int offset, int block_size) nogil
        void dump_mu_d(vector[double] mu, int offset, int block_size) nogil

        void dump_r_d(double* r, int offset, int block_size) nogil
        void dump_r_d(vector[double] r, int offset, int block_size) nogil

        double get_current_slope(int start_row, int stop_row) nogil
        FT get_root_det(int start_row, int end_row) nogil
        FT get_log_det(int start_row, int end_row) nogil
        FT get_slide_potential(int start_row, int end_row, int block_size) nogil

        int babai(vector[ZT] w, vector[FT] v, int start, int dimension) nogil

        const bool enable_int_gram
        const bool enable_row_expo
        const bool enable_transform
        const bool enable_inverse_transform
        const bool row_op_force_long



# LLL

cdef extern from "fplll/lll.h" namespace "fplll":

    cdef cppclass LLLReduction[ZT,FT]:
        LLLReduction(MatGSOInterface[ZT, FT]& m, double delta, double eta, int flags)

        int lll() nogil
        int lll(int kappa_min) nogil
        int lll(int kappa_min, int kappa_start) nogil
        int lll(int kappa_min, int kappa_start, int kappa_end) nogil
        int lll(int kappa_min, int kappa_start, int kappa_end, int size_reduction_start) nogil
        int size_reduction() nogil
        int size_reduction(int kappa_min) nogil
        int size_reduction(int kappa_min, int kappa_end) nogil
        int size_reduction(int kappa_min, int kappa_end, int size_reduction_start) nogil

        int status
        int final_kappa
        int last_early_red
        int zeros
        int n_swaps

    int is_lll_reduced[ZT, FT](MatGSOInterface[ZT, FT]& m, double delta, double eta) nogil


# LLL Wrapper

cdef extern from "fplll/wrapper.h" namespace "fplll":

    cdef cppclass Wrapper:
        Wrapper(ZZ_mat[mpz_t]& b, ZZ_mat[mpz_t]& u, ZZ_mat[mpz_t]& uInv,
                double delta, double eta, int flags)
        int lll() nogil
        int status



# Evaluator

cdef extern from "enumeration_callback_helper.h":
    cdef cppclass PyCallbackEvaluatorWrapper:
        PyCallbackEvaluatorWrapper()
        PyCallbackEvaluatorWrapper(object)


cdef extern from "fplll/enum/evaluator.h" namespace "fplll":

    cdef enum EvaluatorStrategy:
        EVALSTRATEGY_BEST_N_SOLUTIONS
        EVALSTRATEGY_OPPORTUNISTIC_N_SOLUTIONS
        EVALSTRATEGY_FIRST_N_SOLUTIONS


    cdef cppclass Evaluator[FT]:
        Evaluator()

        void eval_sol(const vector[FT]& newSolCoord,
                      const enumf& newPartialDist, enumf& maxDist, long normExp)


        int max_sols
        EvaluatorStrategy strategy
        multimap[FT, vector[FT]] solutions
        size_t sol_count
        vector[pair[FT, vector[FT]]] sub_solutions

        multimap[FP_NR[FT], vector[FP_NR[FT]]].reverse_iterator begin()
        multimap[FP_NR[FT], vector[FP_NR[FT]]].reverse_iterator end()

        int size()
        bool empty()

    cdef cppclass FastEvaluator[FT]:
        FastEvaluator()
        FastEvaluator(size_t nr_solutions, EvaluatorStrategy strategy, bool find_subsolutions)

        void eval_sol(const vector[FT]& newSolCoord,
                      const enumf& newPartialDist, enumf& maxDist, long normExp)

        int max_sols
        EvaluatorStrategy strategy
        multimap[FT, vector[FT]] solutions
        size_t sol_count
        vector[pair[FT, vector[FT]]] sub_solutions

        multimap[FP_NR[FT], vector[FP_NR[FT]]].reverse_iterator begin()
        multimap[FP_NR[FT], vector[FP_NR[FT]]].reverse_iterator end()

        int size()
        bool empty()

    cdef cppclass CallbackEvaluator[FT]:
        CallbackEvaluator()
        CallbackEvaluator(PyCallbackEvaluatorWrapper, void *ctx,
                          size_t nr_solutions, EvaluatorStrategy strategy, bool find_subsolutions)

        void eval_sol(const vector[FT]& newSolCoord,
                      const enumf& newPartialDist, enumf& maxDist, long normExp)

        int max_sols
        EvaluatorStrategy strategy
        multimap[FT, vector[FT]] solutions
        size_t sol_count
        vector[pair[FT, vector[FT]]] sub_solutions

        multimap[FP_NR[FT], vector[FP_NR[FT]]].reverse_iterator begin()
        multimap[FP_NR[FT], vector[FP_NR[FT]]].reverse_iterator end()

        int size()
        bool empty()

    cdef cppclass FastErrorBoundedEvaluator:
        FastErrorBoundedEvaluator()
        FastErrorBoundedEvaluator(int d, Matrix[FP_NR[mpfr_t]] mu, Matrix[FP_NR[mpfr_t]] r, EvaluatorMode eval_mode, size_t nr_solutions, EvaluatorStrategy strategy, bool find_subsolutions)

        void eval_sol(const vector[FP_NR[mpfr_t]]& newSolCoord,
                      const enumf& newPartialDist, enumf& maxDist, long normExp)
        int size()

        int max_sols
        EvaluatorStrategy strategy
        multimap[FP_NR[mpfr_t], vector[FP_NR[mpfr_t]]] solutions
        multimap[FP_NR[mpfr_t], vector[FP_NR[mpfr_t]]].reverse_iterator begin()
        multimap[FP_NR[mpfr_t], vector[FP_NR[mpfr_t]]].reverse_iterator end()


    cdef cppclass ErrorBoundedEvaluator:
        ErrorBoundedEvaluator()
        ErrorBoundedEvaluator(int d, Matrix[FP_NR[mpfr_t]] mu, Matrix[FP_NR[mpfr_t]] r, EvaluatorMode eval_mode, size_t nr_solutions, EvaluatorStrategy strategy, bool find_subsolutions)

        void eval_sol(const vector[FP_NR[mpfr_t]]& newSolCoord,
                      const enumf& newPartialDist, enumf& maxDist, long normExp)
        int size()

        int max_sols
        EvaluatorStrategy strategy
        multimap[FP_NR[mpfr_t], vector[FP_NR[mpfr_t]]] solutions
        multimap[FP_NR[mpfr_t], vector[FP_NR[mpfr_t]]].reverse_iterator begin()
        multimap[FP_NR[mpfr_t], vector[FP_NR[mpfr_t]]].reverse_iterator end()



# Enumeration

cdef extern from "fplll/enum/enumerate.h" namespace "fplll":
    cdef cppclass Enumeration[ZT, FT]:
        Enumeration(MatGSOInterface[ZT, FT]& gso, Evaluator[FT]& evaluator)
        Enumeration(MatGSOInterface[ZT, FT]& gso, FastEvaluator[FT]& evaluator)
        Enumeration(MatGSOInterface[ZT, FP_NR[mpfr_t]]& gso, ErrorBoundedEvaluator& evaluator)
        Enumeration(MatGSOInterface[ZT, FP_NR[mpfr_t]]& gso, FastErrorBoundedEvaluator& evaluator)

        void enumerate(int first, int last, FT& fMaxDist, long maxDistExpo,
                       const vector[FT]& targetCoord,
                       const vector[double]& subTree,
                       const vector[double]& pruning)

        void enumerate(int first, int last, FT& fMaxDist, long maxDistExpo,
                       const vector[FT]& targetCoord,
                       const vector[double]& subTree,
                       const vector[double]& pruning,
                       int dual)

        void enumerate(int first, int last, FT& fMaxDist, long maxDistExpo,
                       const vector[FT]& targetCoord,
                       const vector[double]& subTree,
                       const vector[double]& pruning,
                       int dual,
                       int subtree_reset)

        unsigned long get_nodes(int level)

cdef extern from "fplll/enum/enumerate_ext.h" namespace "fplll":

    ctypedef void extenum_cb_set_config (double *mu, size_t mudim, bool mutranspose, double *rdiag,
                                         double *pruning)

    ctypedef double extenum_cb_process_sol(double dist, double *sol);

    ctypedef void extenum_cb_process_subsol(double dist, double *subsol, int offset);

    ctypedef unsigned long extenum_fc_enumerate(int dim, enumf maxdist,
                                                function[extenum_cb_set_config] cbfunc,
                                                function[extenum_cb_process_sol] cbsol,
                                                function[extenum_cb_process_subsol] cbsubsol,
                                                bool dual, bool findsubsols)

    void set_external_enumerator(function[extenum_fc_enumerate] extenum)
    function[extenum_fc_enumerate] get_external_enumerator()


# SVP

cdef extern from "fplll/svpcvp.h" namespace "fplll":
    int shortest_vector(ZZ_mat[mpz_t]& b,
                        vector[Z_NR[mpz_t]] &sol_coord,
                        SVPMethod method, int flags) nogil

    int shortest_vector_pruning(ZZ_mat[mpz_t]& b, vector[Z_NR[mpz_t]]& sol_coord,
                                const vector[double]& pruning, int flags) nogil

    int shortest_vector_pruning(ZZ_mat[mpz_t]& b, vector[Z_NR[mpz_t]]& sol_coord,
                                vector[vector[Z_NR[mpz_t]]]& auxsol_coord,
                                vector[double]& auxsol_dist, const int max_aux_sols,
                                const vector[double]& pruning, int flags) nogil

    int closest_vector(ZZ_mat[mpz_t] b, vector[Z_NR[mpz_t]] &intTarget,
                       vector[Z_NR[mpz_t]]& sol_coord, CVPMethod method, int flags) nogil



# BKZ

cdef extern from "fplll/bkz_param.h" namespace "fplll":

    cdef cppclass PruningParams:
        double gh_factor
        vector[double] coefficients
        double expectation
        PrunerMetric metric
        vector[double] detailed_cost

        PruningParams()

        @staticmethod
        PruningParams LinearPruningParams(int block_size, int level)

    cdef cppclass Strategy:
        size_t block_size
        vector[PruningParams] pruning_parameters
        vector[size_t] preprocessing_block_sizes

        @staticmethod
        Strategy EmptyStrategy()

        PruningParams get_pruning(double radius, double gh)

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

        double min_success_probability

        int rerandomization_density

        string dump_gso_filename

    vector[Strategy] load_strategies_json(const string &filename) except + nogil
    const string default_strategy_path() nogil
    const string default_strategy() nogil
    const string strategy_full_path(const string &strategy_path) nogil


cdef extern from "fplll/bkz.h" namespace "fplll":

    cdef cppclass BKZReduction[ZT, FT]:

        BKZReduction(MatGSOInterface[ZT, FT] &m, LLLReduction[ZT, FT] &lll_obj, const BKZParam &param) nogil

        int svp_preprocessing(int kappa, int block_size, const BKZParam &param) nogil
        int svp_postprocessing(int kappa, int block_size, const vector[FT] &solution) nogil

        int svp_reduction(int kappa, int block_size, const BKZParam &param, int dual) except + nogil

        int tour(const int loop, int &kappa_max, const BKZParam &param, int min_row, int max_row) except + nogil
        int sd_tour(const int loop, const BKZParam &param, int min_row, int max_row) except + nogil
        int slide_tour(const int loop, const BKZParam &param, int min_row, int max_row) except + nogil

        int hkz(int &kappaMax, const BKZParam &param, int min_row, int max_row) except + nogil

        int bkz()

        void rerandomize_block(int min_row, int max_row, int density) except + nogil

        void dump_gso(const string filename, const string prefix, int append) except + nogil

        int status

        long nodes


    cdef cppclass BKZAutoAbort[ZT, FT]:
        BKZAutoAbort(MatGSOInterface[ZT, FT]& m, int num_rows) nogil
        BKZAutoAbort(MatGSOInterface[ZT, FT]& m, int num_rows, int start_row) nogil

        int test_abort() nogil
        int test_abort(double scale) nogil
        int test_abort(double scale, int max_no_dec) nogil

    void adjust_radius_to_gh_bound[FT](FT& max_dist, long max_dist_expo,
                                        int block_size, FT& root_det_mpfr, double gh_factor) nogil

    FT get_root_det[FT](MatGSOInterface[Z_NR[mpz_t], FT]& m, int start, int end)
    FT get_log_det[FT](MatGSOInterface[Z_NR[mpz_t], FT]& m, int start, int end)
    FT get_sld_potential[FT](MatGSOInterface[Z_NR[mpz_t], FT]& m, int start, int end, int block_size)

    double get_current_slope[FT](MatGSOInterface[Z_NR[mpz_t], FT]& m, int startRow, int stopRow) nogil


# Utility

cdef extern from "fplll/util.h" namespace "fplll":
    void vector_matrix_product(vector[Z_NR[mpz_t]] &result,
                               vector[Z_NR[mpz_t]] &x,
                               const ZZ_mat[mpz_t] &m) nogil




# Pruner

cdef extern from "fplll/pruner/pruner.h" namespace "fplll":

    cdef enum PrunerFlags:
        PRUNER_CVP
        PRUNER_START_FROM_INPUT
        PRUNER_GRADIENT
        PRUNER_NELDER_MEAD
        PRUNER_VERBOSE
        PRUNER_SINGLE
        PRUNER_HALF

    cdef enum PrunerMetric:
        PRUNER_METRIC_PROBABILITY_OF_SHORTEST
        PRUNER_METRIC_EXPECTED_SOLUTIONS

    cdef cppclass Pruner[FT]:
        Pruner(const int n)

        Pruner(const FT enumeration_radius, const FT preproc_cost, const vector[double] &gso_r)

        Pruner(const FT enumeration_radius, const FT preproc_cost, const vector[double] &gso_r,
               const FT target, const PrunerMetric metric, int flags)

        Pruner(const FT enumeration_radius, const FT preproc_cost, const vector[vector[double]] &gso_r)

        Pruner(const FT enumeration_radius, const FT preproc_cost, const vector[vector[double]] &gso_r,
               const FT target, const PrunerMetric metric, int flags)

        void optimize_coefficients(vector[double] &pr)
        void optimize_coefficients_cost_vary_prob(vector[double] &pr)
        void optimize_coefficients_cost_fixed_prob(vector[double] &pr)
        void optimize_coefficients_evec(vector[double] &pr)
        void optimize_coefficients_full(vector[double] &pr)

        double single_enum_cost(const vector[double] &pr, vector[double] *detailed_cost)
        double single_enum_cost(const vector[double] &pr)

        double repeated_enum_cost(const vector[double] &pr)

        double measure_metric(const vector[double] &pr)

        FT gaussian_heuristic()

    void prune[FT](PruningParams &pruning, const double enumeration_radius,
                   const double preproc_cost, const vector[double] &gso_r)

    void prune[FT](PruningParams &pruning, const double enumeration_radius,
                   const double preproc_cost, const vector[double] &gso_r,
                   const double target, const PrunerMetric metric, const int flags)

    void prune[FT](PruningParams &pruning, const double enumeration_radius,
                   const double preproc_cost, const vector[vector[double]] &gso_r)

    void prune[FT](PruningParams &pruning, const double enumeration_radius,
                   const double preproc_cost, const vector[vector[double]] &gso_r,
                   const double target, const PrunerMetric metric, const int flags)

    FT svp_probability[FT](const PruningParams &pruning)
    FT svp_probability[FT](const vector[double] &pr)


# Threads

cdef extern from "fplll/threadpool.h" namespace "fplll":
    int get_threads()
    int set_threads(int th)



# Highlevel Functions

cdef extern from "fplll/fplll.h" namespace "fplll":

    int lll_reduction(ZZ_mat[mpz_t] b, double delta, double eta,
                      LLLMethod method, FloatType float_type,
                      int precision, int flags) nogil
    int lll_reduction(ZZ_mat[mpz_t] b, ZZ_mat[mpz_t] u,
                      double delta, double eta,
                      LLLMethod method, FloatType float_type,
                      int precision, int flags) nogil

    int bkz_reduction(ZZ_mat[mpz_t] *b, ZZ_mat[mpz_t] *u,
                      BKZParam &param, FloatType float_type, int precision) nogil
    int bkz_reduction(ZZ_mat[mpz_t] *b, int block_size, int flags, FloatType float_type, int precision) nogil

    int hkz_reduction(ZZ_mat[mpz_t] b) nogil

    const char* get_red_status_str(int status) nogil
