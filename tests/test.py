from fpylll import *
from copy import copy
import fpylll.contrib.bkz

A = IntegerMatrix(100, 100)
A.randomize("ntrulike", bits=32, q=1456817003)

B = copy(A)
bkz_py = fpylll.contrib.bkz.BKZReduction(B)
param = BKZ.Param(48, flags=BKZ.VERBOSE|BKZ.AUTO_ABORT|BKZ.GH_BND,
                  preprocessing=BKZ.Param(15), pruning=20)
print repr(param)
bkz_py(param)

print

B = copy(A)
bkz_py = fpylll.contrib.bkz.BKZReduction(B)
param = param.new(flags=BKZ.VERBOSE|BKZ.AUTO_ABORT)
print repr(param)
bkz_py(param)
