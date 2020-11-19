# -*- coding: utf-8 -*-

from copy import copy

def float_equals(x, y, epsilon=0.0001):
    """
        Test the equality of x and y up to a relative error epsilon
    """
    if abs(y) < epsilon/2:
        return abs(x) < epsilon/2
    return abs(x / y - 1) < epsilon

def compute_gram(B):
    """
        Compute the Gram matrix of the row-lattice with basis B
    """
    B.transpose()
    Bt = copy(B)
    B.transpose()
    return copy(B * Bt)
