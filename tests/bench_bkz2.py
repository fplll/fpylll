# -*- coding: utf-8 -*-

from fpylll import IntegerMatrix, LLL, GSO
from fpylll import BKZ as fplll_bkz
from fpylll.algorithms.bkz2 import BKZReduction
from time import time
import sys
from fpylll.util import gaussian_heuristic


def benchmark_one(A, b):
    global START, PROBA
    # print
    # print "BLOCKSIZE ", b
    root_vol = 2**(bits/2)
    params = fplll_bkz.Param(block_size=b, strategies="default.json", 
                             flags=0, max_loops=1)
    bkz = BKZReduction(A)
      
    bkz(params=params)
    T = time() - START
    rhf = ((bkz.M.get_r(0, 0)**(.5)) / root_vol)**(1./n)
    r = [bkz.M.get_r(i, i) for i in range(n)]
    hvr = gaussian_heuristic(r[:n/2]) / gaussian_heuristic(r[n/2:])
    gh_factor = r[0] / gaussian_heuristic(r)
    # print "TIME", T
    # print "RHF", rhf
    print bkz.trace.find("enumeration", raise_keyerror=True) 
    print bkz.trace.find("preprocessing", raise_keyerror=True) 
    # print bkz.trace.find("postprocessing", raise_keyerror=True) 
    print bkz.trace.find("pruner", raise_keyerror=True) 
    # print bkz.trace.find("lll", raise_keyerror=True) 
    # print
    # print
    print [b, T, hvr, rhf, gh_factor]
    print


n = 140
bits = 40
A = IntegerMatrix.random(n, "qary", bits=bits, k=n/2, int_type="long")
LLL.reduction(A)

START = time()
for i in range(10, 90, 2):
    benchmark_one(A, i)
