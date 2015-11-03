# -*- coding: utf-8 -*-

from fpylll import IntegerMatrix, MatGSO, LLLReduction, Wrapper
from fpylll import Enumeration as Enum
from fpylll import gso

class BKZ:
    def __init__(self, A):
        """FIXME! briefly describe function

        :param A:
        :returns:
        :rtype:

        """
        if not isinstance(A, IntegerMatrix):
            raise TypeError("Matrix must be IntegerMatrix but got type '%s'"%type(A))

        # run LLL first
        wrapper = Wrapper(A)
        wrapper()

        self.A = A
        self.m = MatGSO(A, flags=gso.ROW_EXPO)
        self.lll_obj = LLLReduction(self.m)

    def __call__(self, block_size):
        """FIXME! briefly describe function

        :param block_size:
        :returns:
        :rtype:

        """
        self.m.discover_all_rows()

        while True:
            clean = self.bkz_loop(block_size, 0, self.A.nrows)
            if clean:
                break

    def bkz_loop(self, block_size, min_row, max_row):
        """FIXME! briefly describe function

        :param block_size:
        :param min_row:
        :param max_row:
        :returns:
        :rtype:

        """
        clean = True
        for kappa in range(min_row, max_row-1):
            bs = min(block_size, max_row - kappa)
            clean &= self.svp_reduction(kappa, bs)
        return clean

    def svp_reduction(self, kappa, block_size):
        """FIXME! briefly describe function

        :param kappa:
        :param block_size:
        :returns:
        :rtype:

        """
        clean = True

        self.lll_obj(0, kappa, kappa + block_size)
        if self.lll_obj.nswaps > 0:
            clean = False

        max_dist, expo = self.m.get_r_exp(kappa, kappa)
        delta_max_dist = self.lll_obj.delta * max_dist

        solution, max_dist = Enum.enumerate(self.m, max_dist, expo, kappa, kappa + block_size, None)

        if max_dist >= delta_max_dist:
            return clean

        nonzero_vectors = len([x for x in solution if x])

        if nonzero_vectors == 1:
            first_nonzero_vector = None
            for i in range(block_size):
                if abs(solution[i]) == 1:
                    first_nonzero_vector = i
                    break

            self.m.move_row(kappa + first_nonzero_vector, kappa)
            self.lll_obj.size_reduction(kappa, kappa + 1)

        else:
            d = self.m.d
            self.m.create_row()
            self.m.row_op_begin(d, d + 1)
            for i in range(block_size):
                self.m.row_addmul(d, kappa + i, solution[i])
            self.m.row_op_end(d, d + 1)
            self.m.move_row(d, kappa)
            self.lll_obj(kappa, kappa, kappa + block_size + 1)
            self.m.move_row(kappa + block_size, d)
            self.m.remove_last_row()

        return False
