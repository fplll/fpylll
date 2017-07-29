from fpylll import IntegerMatrix, BKZ, Pruning
from fpylll.fplll.bkz_param import Strategy


def test_linear_pruning():
    A = IntegerMatrix.random(25, "qary", k=15, q=127)
    block_size  = 10
    preprocessing = 3
    strategies = [Strategy(i) for i in range(5)]

    for b in range(5, block_size+1):
        strategies.append(Strategy(b, [preprocessing], [Pruning.LinearPruningParams(b, 2)]))

    param = BKZ.Param(block_size=block_size, strategies=strategies)
    BKZ.reduction(A, param)
