# -*- coding: utf-8 -*-

from fpylll import IntegerMatrix, LLL, GSO
from fpylll import BKZ as fplll_bkz
from fpylll import Enumeration
from fpylll import EnumerationError
from fpylll.algorithms.bkz2_otf_subsol import BKZReduction
from fpylll.fplll.pruner import prune
from fpylll.fplll.pruner import Pruning
from math import log, sqrt
import pickle

from math import floor 

from time import time
import sys
from fpylll.util import gaussian_heuristic

start_dim = 80
bs = int(sys.argv[1])

try:
    repeat = int(sys.argv[2])
except:
    repeat = 1

NPS = 2**24


def IntegerMatrix_to_long(A):
    n = A.nrows
    AA = IntegerMatrix(n, n, int_type="long")
    for i in xrange(n):
        for j in xrange(n):
            AA[i, j] = A[i, j]
    return AA


def print_basis_stats(M, n):
    r = [M.get_r(i, i) for i in range(n)]
    gh = gaussian_heuristic(r)
    lhvr = log(gaussian_heuristic(r[:n/2])) - log(gaussian_heuristic(r[n/2:]))
    print "lhrv = %.4f, r[i]/gh"%lhvr,
    for i in range(20):
        print "%.3f"%(r[i]/gh), 
    print
    return


def insert_sub_solutions(bkz_obj, sub_solutions):
    M = bkz_obj.M
    l = len(sub_solutions)
    n = M.d
    for (a, vector) in sub_solutions:
        M.create_row()
        if len(vector)==0:      # No subsolution at this index. Leaving a 0 vector
            continue 
        with M.row_ops(M.d-1, M.d):
            for i in range(n):                    
                M.row_addmul(M.d-1, i, vector[i])    

    for k in reversed(range(l)):
        M.move_row(M.d-1, k)

    bkz_obj.lll_obj()

    for i in range(l):
        M.remove_last_row()

    return

def save_buggy_instance(a,b,c):
    pickle.dump((a, b, c), open( "buugy_inst.pkl", "wb" ) )
    for i in range(20):
        print "TOTAL_TIME -- BUUUUG"
    exit(1)


def enum_trial(bkz_obj, preproc_cost, gh_factor=1.1):
    n = bkz_obj.A.nrows

    r = [bkz_obj.M.get_r(i, i) for i in range(0, n)]       
    gh = gaussian_heuristic(r)
    radius = max(r[0] * .99, gh * gh_factor)
    PRUNE_START = time()
    pruning = prune(radius, NPS * preproc_cost, [r], 10, 
                    metric="solutions", float_type="dd",
                    flags=Pruning.GRADIENT)
    PRUNE_TIME = time() - PRUNE_START
    print "Pruning time %.4f"%PRUNE_TIME
    ENUM_START = time()
    enum_obj = Enumeration(bkz_obj.M, sub_solutions=True)

    try:
        print "Enum  ... (Expecting %.5f solutions)"%(pruning.expectation), 
        enum_obj.enumerate(0, n, radius, 0, pruning=pruning.coefficients)
    except EnumerationError:
        pass

    ENUM_TIME = time() - ENUM_START
    print " \t\t\t\t\t\t TIME = %.2f"%ENUM_TIME

    zeros = 0
    print "subsolutions : r[i]/gh", 
    for (a, b) in enum_obj.sub_solutions[:20]:
        print "%.3f"%abs(a/gh),

    insert_sub_solutions(bkz_obj, enum_obj.sub_solutions[:n/4])
    return 


def asvp(A, bs, gh_factor=1.1):
    n = A.nrows
    bkz = BKZReduction(A)
    bkz.lll_obj()
    bkz.randomize_block(0, n, density=n/4)
    bkz.lll_obj()
    r = [bkz.M.get_r(i, i) for i in range(n)]
    gh = gaussian_heuristic(r)

    trials = 0
    while r[0] > gh * gh_factor:
        r = [bkz.M.get_r(i, i) for i in range(n)]

        print
        BKZ_START = time()
        # print_basis_stats(bkz.M, n)
        print "BKZ-[%d .. %d]  ... \t\t "%(bs-20, bs),
        for lbs in [bs - 20, bs - 10, bs]:
            params = fplll_bkz.Param(block_size=lbs, max_loops=1,
                                     min_success_probability=.01) #, flags=fplll_bkz.BOUNDED_LLL)
            bkz(params=params)
            bkz.lll_obj()
        r = [bkz.M.get_r(i, i) for i in range(n)]
        BKZ_TIME = time() - BKZ_START
        print "  \t\t\t\t\t\t\t TIME = %.2f"%BKZ_TIME
        print_basis_stats(bkz.M, n)

        enum_trial(bkz, BKZ_TIME, gh_factor=gh_factor)
        print
        r = [bkz.M.get_r(i, i) for i in range(n)]
        trials += 1

    print "Finished !"
    print_basis_stats(bkz.M, n)
    return trials


START = time()
for dim in range(start_dim, 130, 2):
    for r in range(repeat):
        A_pre = IntegerMatrix.from_file("svpchallenge/svpchallengedim%dseed0.txt"%dim)
        print "---------------------", A_pre.nrows
        ASVP_START = time()
        LLL.reduction(A_pre)

        # bs = int(floor(dim * bs_rat))

        A = IntegerMatrix.from_matrix(A_pre, int_type="long")

        trials = asvp(A, bs, gh_factor=(1.05**2))
        ASVP_TIME = time() - ASVP_START

        print "\n\n Challenge %d Solved"%dim
        print A[0]
        print "SUMMARY", {"dim": dim, "bs": bs, "time": ASVP_TIME, "trials": trials}

