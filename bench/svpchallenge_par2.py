# -*- coding: utf-8 -*-

from copy import copy
from multiprocessing import Process
from time import sleep, clock, time
from random import randint

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

import sys
from fpylll.util import gaussian_heuristic

start_dim = 80
bs_diff = int(sys.argv[1])
cores = int(sys.argv[2])

BS_RANGE = 10

NPS = 2**24


def copy_to_IntergerMatrix_long(A):
    n = A.nrows
    AA = IntegerMatrix(n, n, int_type="long")
    for i in xrange(n):
        for j in xrange(n):
            AA[i, j] = A[i, j]
    return AA


def insert_in_IntergerMatrix(A, v):
    n = A.nrows
    AA = IntegerMatrix(n + 1, n, int_type="long")
    for j in xrange(n):
        AA[0, j] = v[j]
        for i in xrange(n):
            AA[i + 1, j] = A[i, j]

    LLL.reduction(AA)
    for j in xrange(n):
        for i in xrange(n):
            A[i, j] = AA[i + 1, j]

    del AA


def print_basis_stats(M, n):
    r = [M.get_r(i, i) for i in range(n)]
    gh = gaussian_heuristic(r)
    lhvr = log(gaussian_heuristic(r[:n/2])) - log(gaussian_heuristic(r[n/2:]))
    print "lhrv = %.4f, r[i]/gh"%lhvr,
    for i in range(20):
        print "%.3f"%(r[i]/gh), 
    print
    return


def insert_sub_solutions(bkz, sub_solutions):
    M = bkz.M
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

    bkz.lll_obj()

    for i in range(l):
        M.remove_last_row()

    return


def enum_trial(bkz, preproc_cost, radius):
    n = bkz.A.nrows

    r = [bkz.M.get_r(i, i) for i in range(0, n)]       
    gh = gaussian_heuristic(r)

    PRUNE_START = time()
    pruning = prune(radius, NPS * preproc_cost, [r], 10, 
                    metric="solutions", float_type="dd",
                    flags=Pruning.GRADIENT|Pruning.NELDER_MEAD)
    PRUNE_TIME = time() - PRUNE_START    
    ENUM_START = time()
    enum_obj = Enumeration(bkz.M, sub_solutions=True)
    success = False
    try:        
        enum_obj.enumerate(0, n, radius, 0, pruning=pruning.coefficients)
        success = True
    except EnumerationError:
        pass
    print "Pruning time %.4f"%PRUNE_TIME
    print "Enum  ... (Expecting %.5f solutions)"%(pruning.expectation), 

    ENUM_TIME = time() - ENUM_START
    print " \t\t\t\t\t\t TIME = %.2f"%ENUM_TIME

    zeros = 0
    print "subsolutions : r[i]/gh", 
    for (a, b) in enum_obj.sub_solutions[:20]:
        print "%.3f"%abs(a/gh),
    print 

    insert_sub_solutions(bkz, enum_obj.sub_solutions[:n/4])
    return success


def svp_improve_trial(filename, bs):
    A, _ = pickle.load(open(filename, 'rb'))
    n = A.nrows
    bkz = BKZReduction(A)
    bkz.lll_obj()
    r = [bkz.M.get_r(i, i) for i in range(n)]
    print filename, "before BKZ",
    print_basis_stats(bkz.M, n)
    gh = gaussian_heuristic(r)

    BKZ_START = time()
    for lbs in range(30, bs - 10, 2) + [bs]:
        params = fplll_bkz.Param(block_size=lbs, max_loops=1,
                                 min_success_probability=.01)
        bkz(params=params)
        bkz.lll_obj()

    r = [bkz.M.get_r(i, i) for i in range(n)]

    BKZ_TIME = time() - BKZ_START
    print filename, "BKZ-[%d .. %d]  ... \t\t "%(30, bs),
    print "  \t\t\t\t\t\t\t TIME = %.2f"%BKZ_TIME
    print filename, 
    print_basis_stats(bkz.M, n)
    success = enum_trial(bkz, BKZ_TIME, r[0]*.99)
    print
    pickle.dump((A, success), open(filename, 'wb'))
    return success


class SVPool:
    def __init__(self, max_len, copies=1):
        self.max_len = max_len
        self.copies = copies
        self.data = []

    def push(self, v):
        norm = sum([x*x for x in v])
        for i in range(self.copies):
            self.data += [copy((norm, v))]
        if len(self.data)> self.max_len:
            self.data.sort()

    def pop(self):
        l = len(self.data)
        if l==0:
            return None
        i = randint(0, l-1)
        res = copy(self.data[i][1])
        del self.data[i]
        return res

POOL_SIZE = 8 * cores
POOL_COPIES = 1 + cores/4
POOL_INSERT = 2

def interacting_parrallel_asvp(A, bs_max, goal, cores):
    n = A.nrows

    trials = cores * [0]
    As = cores * [None]

    for i in range(cores):
        As[i] = copy_to_IntergerMatrix_long(A)
        bkz = BKZReduction(As[i])
        bkz.randomize_block(0, n, density=n/4)
        del bkz

    sv_pool = SVPool(POOL_SIZE, copies=POOL_COPIES)
    workers = cores*[None]
    over = False

    while not over:
        sleep(.1)
        for i in range(cores):

            if workers[i] is None:
                v = sv_pool.pop()
                if v is not None:
                    print "POPPED"
                    insert_in_IntergerMatrix(As[i], v)

                bsi = bs_max - 20
                bsi += min(20, 2*trials[i])
                bsi -= 2*randint(0, BS_RANGE/2)
                pickle.dump((As[i], False), open("%d.tmp"%i, 'wb'))
                workers[i] = Process(target=svp_improve_trial, args=("%d.tmp"%i, bsi))
                workers[i].start()

            if (workers[i] is not None) and (not workers[i].is_alive()):
                As[i], success = pickle.load(open("%d.tmp"%i, 'rb'))
                print "SUCC: ", success
                print
                if success:
                    sv_pool.push([x for x in As[i][0]])
                workers[i] = None
                trials[i] += 1
                norm = sum([x*x for x in As[i][0]])
                if norm < goal:
                    print "SVP-%d SOLUTION :"%n, As[i][0]
                    over = True
                    break


    for w in [w for w in workers if w is not None]:
        w.terminate()

    while True:
        sleep(.1)
        some_alive = False
        for w in [w for w in workers if w is not None]:
            some_alive |= w.is_alive()
        if not some_alive:
            return

print cores

START = time()
for dim in range(start_dim, 130, 2):
    A_pre = IntegerMatrix.from_file("svpchallenge/svpchallengedim%dseed0.txt"%dim)
    print "---------------------", A_pre.nrows
    ASVP_START = time()
    LLL.reduction(A_pre)

    bs_max = dim - bs_diff
    A = IntegerMatrix.from_matrix(A_pre, int_type="long")

    bkz = BKZReduction(A)
    bkz.lll_obj()
    r = [bkz.M.get_r(i, i) for i in range(dim)]
    goal = (1.05)**2 * gaussian_heuristic(r)

    interacting_parrallel_asvp(A, bs_max, goal, cores)
    ASVP_TIME = time() - ASVP_START

    print "\nSUMMARY", {"dim": dim, "bs_range": (bs_max - BS_RANGE, bs_max), "time": ASVP_TIME}

