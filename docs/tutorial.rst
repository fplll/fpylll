.. role:: math(raw)
   :format: html latex
..

:orphan:

.. role:: raw-latex(raw)
   :format: latex
..

.. _tutorial:

Tutorial
========


Matrix generators
-----------------

::

    >>> from fpylll import IntegerMatrix, FPLLL
    >>> FPLLL.set_random_seed(1337)
    >>> A = IntegerMatrix(9, 10)
    >>> A.randomize("intrel", bits=10)

Matrix :math:`A` is a (random) knapsack type matrix. That is of the form  :math:`[ {\bf a} | I_n]`, where :math:`{\bf a}` is a column vector of dimension :math:`n`, and :math:`I_n` the :math:`n`-dimensional identity matrix. Giving

::

    >>> print(A)
    [  50 1 0 0 0 0 0 0 0 0 ]
    [ 556 0 1 0 0 0 0 0 0 0 ]
    [   5 0 0 1 0 0 0 0 0 0 ]
    [ 899 0 0 0 1 0 0 0 0 0 ]
    [ 383 0 0 0 0 1 0 0 0 0 ]
    [ 846 0 0 0 0 0 1 0 0 0 ]
    [ 771 0 0 0 0 0 0 1 0 0 ]
    [ 511 0 0 0 0 0 0 0 1 0 ]
    [ 734 0 0 0 0 0 0 0 0 1 ]



Also, the following types of matrices are supported,

::

    >>> from fpylll import FPLLL
    >>> from copy import copy
    >>> b = 10
    >>> p = 521 # prime
    >>> FPLLL.set_random_seed(1337)
    >>> A = IntegerMatrix(6,6)
    >>> B = copy(A)
    >>> C = copy(A)
    >>> D = copy(A)
    >>> A.randomize("uniform", bits=b)
    >>> B.randomize("ntrulike", bits=b, q=p)
    >>> C.randomize("ntrulike2", bits=b, q=p)
    >>> D.randomize("qary", bits=b, k=3)


For instance::

    >>> print(D)
    [ 1 0 0 858 790 620 ]
    [ 0 1 0  72 832 133 ]
    [ 0 0 1 263 121 724 ]
    [ 0 0 0 877   0   0 ]
    [ 0 0 0   0 877   0 ]
    [ 0 0 0   0   0 877 ]


For a user defined matrix we use the method ``from_matrix``::

    >>> A = IntegerMatrix.from_matrix([[1,2,3,4],[30,4,4,5],[1,-2,3,4],[0,0,1,0]])

Gram-Schmidt tools
-------------------

To compute the Gram-Schimdt form of the matrix :math:`{\bf A}`, we use the GSO class::

    >>> from fpylll import GSO
    >>> A = IntegerMatrix.from_matrix([[1,2,3,4],[30,4,4,5],[1,-2,3,4],[0,0,1,0]])
    >>> M = GSO.Mat(A)

To write a vector :math:`v` as a linear combination of the GS-basis of RowSp(:math:`A`)::

    >>> A = IntegerMatrix.from_matrix([[1,2,3,4],[30,4,4,5],[1,-2,3,4],[0,0,1,0]])
    >>> M = GSO.Mat(A)
    >>> _ = M.update_gso()
    >>> v = (1,2,5,5)
    >>> v_from_canonical = M.from_canonical(v)
    >>> print(v_from_canonical) # doctest: +ELLIPSIS
    (1.3333333333333..., -0.01301973960520..., 0.1949374454466..., 1.2521739130434...)
    >>> v_back_to_canonical = tuple([int(round(v_)) for v_ in M.to_canonical(v_from_canonical)])
    >>> print(v_back_to_canonical)
    (1, 2, 5, 5)
    >>> # the dimension of the GS-matrix :
    ... print(M.d)
    4


We can then compute the inner product :math:`r_{i,j} = \langle {\bf b}_i, {\bf b}^{*}_j \rangle` and the coefficient :math:`\mu_{i,j} = \langle {\bf b}_i, {\bf b}^*_j \rangle / ||{\bf b}^*_j||^2`
(for any :math:`i,j`, here :math:`i=2`, :math:`j=3`)

::

    >>> i = 3; j = 2;
    >>> print(M.get_r(i,j)) # doctest: +ELLIPSIS
    0.810079798...
    >>> print(M.get_mu(i,j))  # doctest: +ELLIPSIS
    0.0584569876...

To compute the determinant of :math:`{\bf A}`, compute either its :math:`\sqrt[n]{~}` or its :math:`\log`

::

    >>> start_row = 0
    >>> stop_row = -1
    >>> root_det_A = M.get_root_det(start_row, stop_row)
    >>> log_det_A = M.get_log_det(start_row, stop_row)
    >>> print(root_det_A) # root_det_A = det(A)^(1/n) doctest: +ELLIPSIS
    21.44761058...
    >>> print(log_det_A) # log_det_A = exp(det(A)) in base e doctest: +ELLIPSIS
    12.26245297...

Lattice reduction tools
------------------------

To compute the LLL reduced matrix of :math:`{\bf A}`

::

    >>> from fpylll import LLL
    >>> FPLLL.set_random_seed(1337)
    >>> A.randomize("qary", bits=10, k=3)
    >>> A_original = copy(A)
    >>> A_lll = LLL.reduction(A)
    >>> print(A_lll)
    [  -1  9 -5  -3 ]
    [  12 -2  7 -17 ]
    [ -18  3 16  -1 ]
    [   4 17 20  12 ]

To test if a matrix is LLL-reduced

::

    >>> print(LLL.is_reduced(A_original)) # a uniform matrix is usually not LLL-reduced
    False
    >>> print(LLL.is_reduced(A_lll))
    True

For the BKZ reduction of :math:`{\bf A}` with blocksize say 3 (without pruning),

::

    >>> from fpylll import BKZ
    >>> block_size = 3
    >>> FPLLL.set_random_seed(1337)
    >>> A.randomize("qary", bits=10, k=3)
    >>> A_bkz = BKZ.reduction(A, BKZ.Param(block_size))
    >>> print(A_bkz)
    [  -1  9 -5  -3 ]
    [  12 -2  7 -17 ]
    [ -18  3 16  -1 ]
    [   4 17 20  12 ]

If we want to use pruning we can use the default pruning of fplll [GNR10]_.

::

    >>> from fpylll import BKZ
    >>> param = BKZ.Param(block_size = block_size, strategies = BKZ.DEFAULT_STRATEGY)
    >>> bkz_reduced = BKZ.reduction(A, param)

SVP and CVP tools
-----------------

To use Babai's Nearest Plane algorithm on the target vector :math:`v` with basis :math:`{\bf A}`,
use it from the GSO tool detailed above

::
    >>> FPLLL.set_random_seed(1337)
    >>> A = LLL.reduction(IntegerMatrix.random(5, "qary", bits=10, k=3))
    >>> M = GSO.Mat(A)
    >>> _ = M.update_gso()
    >>> w = M.babai([1, 17, -3, -75, 102])
    >>> A.multiply_left(w)
    (-4, 16, -5, -78, 97)

To compute the norm of a shortest vector of the lattice generated by the rows of the matrix :math:`{\bf A}` we use the ``shortest_vector`` method of the SVP class, and measure the first row of the resulting matrix :math:`{\bf A}`

::

    >>> from fpylll import SVP
    >>> from numpy import linalg as LA
    >>> import numpy as np
    >>> SVP.shortest_vector(A)
    (2, -2, 7, 4, -1)
    >>> print(A[0])
    (2, -2, 7, 4, -1)
    >>> A[0].norm()
    8.602325267042627

For the Closest Vector Problem, fplll (and so fpylll) uses enumeration::

    >>> from fpylll import CVP
    >>> A = IntegerMatrix.from_matrix([[1,2,3,4],[30,4,4,5],[1,-2,3,4]])
    >>> t = (1, 2, 5, 5)
    >>> v0 = CVP.closest_vector(A, t)
    >>> v0
    (1, 2, 3, 4)

In fact the following code was executed::

    >>> from fpylll.fplll.gso import MatGSO
    >>> from fpylll.fplll.enumeration import Enumeration
    >>> M = MatGSO(A)
    >>> _ = M.update_gso()
    >>> E = Enumeration(M)
    >>> _, v2 = E.enumerate(0, A.nrows, 5, 40, M.from_canonical(t))[0]
    >>> v3 = IntegerMatrix.from_iterable(1, A.nrows, map(lambda x: int(x), v2))
    >>> v1 = v3*A
    >>> print(v1)
    [ 1 2 3 4 ]

Further examples
----------------

More specific examples can be found in:

* :doc:`example-gauss-circle-problem`
* :doc:`example-linear-diophantine-equations`
* :doc:`example-custom-pruning`

References
----------

.. [MV] D. Micciancio, P. Voulgaris,  Faster exponential time algorithms for the shortest vector problem. In: SODA 2010, pp. 1468--1480 (2010).
.. [GNR10] Nicolas Gama, Phong Q. Nguyen, and Oded Regev. 2010. Lattice enumeration using extreme pruning. In Proceedings of the 29th Annual international conference on Theory and Applications of Cryptographic Techniques (EUROCRYPT'10), Henri Gilbert (Ed.). Springer-Verlag, Berlin, Heidelberg, 257-278. DOI=http://dx.doi.org/10.1007/978-3-642-13190-5_13
