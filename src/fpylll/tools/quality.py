# -*- coding: utf-8 -*-
"""

"""

from math import log, exp
from collections import OrderedDict
from fpylll.util import gaussian_heuristic


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

    :param M: A MatGSO object.

    :example:

        >>> from fpylll import IntegerMatrix, GSO, LLL, FPLLL
        >>> FPLLL.set_random_seed(1337)
        >>> A = IntegerMatrix.random(100, "qary", bits=30, k=50)
        >>> _ = LLL.reduction(A)
        >>> M = GSO.Mat(A); _ = M.update_gso()

        >>> from fpylll.tools.quality import basis_quality
        >>> from fpylll.tools.bkz_stats import pretty_dict
        >>> str(pretty_dict(basis_quality(M)))
        '{"r_0":   2^34.0,  "r_0/gh": 9.389811,  "rhf": 1.020530,  "/": -0.08550,  "hv/hv": 2.940943}'

    """

    d = M.d

    ret = OrderedDict()

    r = [M.get_r(i, i) for i in range(d)]

    log_volume = sum(log(r_)/2 for r_ in r)

    lhs = sum(log(r_)/2 for r_ in r[:d//2])
    rhs = sum(log(r_)/2 for r_ in r[d//2 + (d%2):])

    ret["r_0"] = r[0]
    ret["r_0/gh"] = r[0]/gaussian_heuristic(r)
    ret["rhf"] = exp((log(r[0])/2.0 - log_volume/d)/d)
    ret['/']   = M.get_current_slope(0, d)
    ret["hv/hv"] = exp((lhs - rhs)/d)

    return ret
