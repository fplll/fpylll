from fpylll import *

A = IntegerMatrix(50, 50)
A.randomize("ntrulike", bits=50, q=127)
lll_reduction(A)

from examples.bkz import BKZ
