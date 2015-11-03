from fpylll import *

from copy import copy

A = IntegerMatrix(50, 50)
A.randomize("ntrulike", bits=50, q=127)

B = copy(A)
bkz_reduction(B, 20)
