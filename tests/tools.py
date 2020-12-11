# -*- coding: utf-8 -*-

from copy import copy


def compute_gram(B):
    """
        Compute the Gram matrix of the row-lattice with basis B
    """
    B.transpose()
    Bt = copy(B)
    B.transpose()
    return copy(B * Bt)
