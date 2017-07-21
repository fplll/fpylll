.. role:: math(raw)
   :format: html latex
..

.. role:: raw-latex(raw)
   :format: latex
..


Small solutions to a linear diophantine equation
================================================

Say we want find a small solution to a linear diophantine equation `\sum_{i=1}^{n}a_ix_i=a_0`.
In general Euclidean algorithm will provide a solution in polynomial time. This method does not provide small solution in general. We can use the lattice from [1]_ to attack this problem.
Let A be the basis:

.. math::

   A = \begin{bmatrix}
   1 & 0 & 0 & \cdots & 0 & 0 & N_2a_1 \\
   0 & 1 & 0 & \cdots & 0 & 0 & N_2a_2 \\
   \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
   0 & 0 & 0 & \cdots & 1& 0 &N_2a_n \\
   0 & 0 & 0 & \cdots & 0 & N_1 &-N_2a_0
   \end{bmatrix}

where `N_1`, `N_2` are some positive integers. Say `(x_1,x_2,...,x_n,x_{n+1},x_{n+2})` is a row of the LLL-reduced matrix of A. If `x_{n+1}=N_1, x_{n+2}=0,` then `(x_1,...,x_n)` is a solution of the linear equation.
Say `{\bf a} = (1124, 1799, 1151, 1979, 1799, 1625, 1077, 1666, 1438, 1739)`, `a_0=22833`, `N_1=100`, `N_2=1000`. We use numpy matrices to define the previous matrix.

::

  >>> from fpylll import *
  >>> import numpy as np
  >>> from numpy import matrix
  >>> from numpy import linalg as LA
  >>> N1 = 100
  >>> N2 = 10000
  >>> a = [1124, 1799, 1151, 1979, 1799, 1625, 1077, 1666, 1438, 1739]
  >>> a0 = 22833
  >>> n = len(a)
  >>> zero_matrix = matrix(np.zeros(n))
  >>> L = matrix([[0 for i in range(0, n)], [N2*a[i] for i in range(0, n)]])
  >>> L = L.transpose() #the two last columns
  >>> M =  np.identity(n)
  >>> Mb = np.hstack([M, L])
  >>> last_line = matrix(np.hstack((np.zeros(n), np.array([N1, -N2*a0]))))
  >>> Mb = np.vstack((Mb, last_line))

Note that before we apply LLL to Mb we need to convert it to fpyll IntegerMatrix.

::

  >>> print LLL.reduction(IntegerMatrix.from_matrix([map(int, row) for row in Mb.tolist()]))
  [  0 -1  0  0  1  0  0  0  0 0   0      0 ]
  [  0  1  0  0  0  0  1  0 -2 0   0      0 ]
  [ -1  0 -1 -1  0  0  1  0  1 1   0      0 ]
  [ -1 -1  0  1 -1  0  1  1  0 0   0      0 ]
  [  1 -1  0  0 -1  1  1 -1  1 0   0      0 ]
  [  0  0  0  0  0  1 -2  1 -2 1   0      0 ]
  [  0  0  0  0  0 -2  0 -1  1 2   0      0 ]
  [ -1 -1  2  0  0  1 -1 -1  0 1   0      0 ]
  [ -2  1 -2  3  0  2  0 -3 -1 0   0      0 ]
  [  1  2  0  1  2  3  1  1  1 2 100      0 ]
  [  1  0  0  1  0  0  0 -1 -1 0   0 -10000 ]

So a small solution is `{\bf v} = ( 1,  2,  0,  1,  2,  3,  1,  1,  1, 2 ),` with norm

::

  >>> LA.norm(np.array([1,  2,  0,  1,  2,  3,  1,  1,  1, 2]))
  5.0990195135927845

.. [1] K. Aardal, C. Hurkens, A. Lenstra, Solving a linear Diophantine equation with lower and upper bounds on the variables. Integer programming and combinatorial optimization LNCS 1412, p.229–242, 1998.
