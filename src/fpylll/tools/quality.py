# -*- coding: utf-8 -*-
"""

"""

from math import log, exp
from collections import OrderedDict
from fpylll.util import gaussian_heuristic


def get_current_slope(r, start_row=0, stop_row=-1):
    """
    A Python re-implementation of ``MatGSO.get_current_slope``.

        >>> from fpylll import IntegerMatrix, GSO, LLL, FPLLL
        >>> FPLLL.set_random_seed(1337)
        >>> A = IntegerMatrix.random(100, "qary", bits=30, k=50)
        >>> _ = LLL.reduction(A)
        >>> M = GSO.Mat(A); _ = M.update_gso()
        >>> from fpylll.tools.quality import get_current_slope
        >>> M.get_current_slope(0, 100)  # doctest: +ELLIPSIS
        -0.083173398...
        >>> get_current_slope(M.r(), 0, 100) # doctest: +ELLIPSIS
        -0.083173398...

    """
    x = [log(r[i]) for i in range(start_row, stop_row)]
    n = stop_row - start_row
    i_mean = (n - 1) * 0.5 + start_row
    x_mean = sum(x)/n
    v1, v2 = 0.0, 0.0

    for i in range(start_row, stop_row):
        v1 += (i - i_mean) * (x[i] - x_mean)
        v2 += (i - i_mean) * (i - i_mean)
    return v1 / v2


def basis_quality(M):
    r"""
    Return a dictionary with various expressions of quality of the basis corresponding to ``M``.

    Let `|b_i^*|` be the norm of the `i`-th Gram-Schmidt vector.  Let `Λ` be the lattice spanned by
    the basis of dimension `d`.

        - ``r_0`` - `|b_0|^2`

        - ``\`` - the slope of `\log(|b_i^*|)`

        - ``rhf`` - the root-Hermite factor `|b_0|/\Vol(Λ)^{1/d}` also written as `\delta_0`

        - ``hv/hv`` - the dth-root of the fraction of the first and second half-volumes, i.e. the
          dth-root of `∏_{i=0}^{d/2-1} |b_i|/∏_{i=d/2}^{d-1} |b_i|`.  If `d` is odd, the length
          `|b_{d//2}|` is ignored.

        - ``r_0/gh`` - `|b_0|/GH` where `GH = Γ(d/2+1)^{1/d}/π^{1/2} ⋅ \Vol(Λ)^{1/d}` is the Gaussian
          Heuristic for the shortest vector.

    :param M: A MatGSO object or a vector of squared Gram-Schmidt norms.

    Example:

        >>> from fpylll import IntegerMatrix, GSO, LLL, FPLLL
        >>> FPLLL.set_random_seed(1337)
        >>> A = IntegerMatrix.random(100, "qary", bits=30, k=50)
        >>> _ = LLL.reduction(A)
        >>> M = GSO.Mat(A); _ = M.update_gso()

        >>> from fpylll.tools.quality import basis_quality
        >>> from fpylll.tools.bkz_stats import pretty_dict
        >>> str(pretty_dict(basis_quality(M)))
        '{"r_0":   2^35.3,  "r_0/gh": 8.564671,  "rhf": 1.020061,  "/": -0.08317,  "hv/hv": 2.832300}'
        >>> str(pretty_dict(basis_quality(M.r())))
        '{"r_0":   2^35.3,  "r_0/gh": 8.564671,  "rhf": 1.020061,  "/": -0.08317,  "hv/hv": 2.832300}'

    """

    try:
        d = M.d
        r = [M.get_r(i, i) for i in range(d)]
    except AttributeError:
        d = len(M)
        r = M

    ret = OrderedDict()

    log_volume = sum(log(r_)/2 for r_ in r)

    lhs = sum(log(r_)/2 for r_ in r[:d//2])
    rhs = sum(log(r_)/2 for r_ in r[d//2 + (d%2):])

    ret["r_0"] = r[0]
    ret["r_0/gh"] = r[0]/gaussian_heuristic(r)
    ret["rhf"] = exp((log(r[0])/2.0 - log_volume/d)/d)
    try:
        ret['/'] = M.get_current_slope(0, d)
    except AttributeError:
        ret["/"] = get_current_slope(M, 0, d)
    ret["hv/hv"] = exp((lhs - rhs)/d)

    return ret
