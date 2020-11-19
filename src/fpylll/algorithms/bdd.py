from fpylll import IntegerMatrix, GSO, LLL, BKZ
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2
from fpylll.util import gaussian_heuristic
from fpylll.tools.bkz_simulator import simulate


def closest_vector(A, target, error_norm=None, float_type="d", **kwds):
    try:
        int_type = A.int_type
    except AttributeError:
        int_type = "mpz_t"

    # Absorb whatever is thrown at us
    A = IntegerMatrix.from_matrix(A, int_type=int_type)
    A = LLL.reduction(A)

    d = A.nrows + 1

    if error_norm is None:
        M = GSO.Mat(A, float_type=float_type)
        M.update_gso()
        error_norm = 0.99 * gaussian_heuristic(M.r())**(.5)

    tau = max(int(round(error_norm/d**(.5))), 1)

    B = IntegerMatrix(A.nrows + 1, A.ncols + 1, int_type=A.int_type)
    for i in range(A.nrows):
        for j in range(A.ncols):
            B[i, j] = A[i, j]
        B[-1, i] = target[i]
    B[-1, -1] = tau

    B = LLL.reduction(B)
    M = GSO.Mat(B, float_type=float_type)
    M.update_gso()

    found = False
    for k in range(2, d)[::-1]:
        param = BKZ.EasyParam(k, **kwds)
        if k >= len(param.strategies):
            continue
        r = simulate(list(M.r()), param)[0]
        if k * (error_norm** 2 + tau **2) / d < r[-k]:
            found = k
        else:
            break

    if found >= len(param.strategies) - 1:
        raise NotImplementedError("No strategy for block size %d found."%found)

    if not found:
        raise RuntimeError("Cannot find block size to solve instance")

    for k in list(range(8, found, 8)) + [found]:
        BKZ2(M)(BKZ.EasyParam(k, **kwds))
        if abs(M.B[0, -1]) == tau and M.get_r(0, 0) <= (1.01 * error_norm)**2:
            print(k)
            break

    return tuple(M.B[0])[:-1]
