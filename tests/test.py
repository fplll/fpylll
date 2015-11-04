from fpylll import *

from copy import copy

A = IntegerMatrix(100, 100)
A.randomize("ntrulike", bits=200, q=127)

B = copy(A)
bkz_reduction(B, 20, verbose=True)
