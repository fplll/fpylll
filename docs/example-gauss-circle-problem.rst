.. _example-gauss-circle-problem:

.. role:: math(raw)
   :format: html latex
..

:orphan:

.. role:: raw-latex(raw)
   :format: latex
..

Gauss Circle Problem
====================

Using enumeration we can study the Gauss circle problem [1]_, i.e. the problem of finding the number `N_R=|\{(x,y)\in {\mathbb{Z}}^2 : x^2+y^2\leq R^2 \}|`. Gauss proved that `N_R=\pi R^2 + Err(R),` with `|Err(R)|\leq 2\sqrt{2}\pi R`. The following code will compute all the points in the set `\{{(x,y)\in \mathbb{Z}}^2 : x^2+y^2\leq R^2\}`:

::

  >>> from fpylll.fplll.gso import MatGSO
  >>> from fpylll.fplll.integer_matrix import IntegerMatrix
  >>> from fpylll import FPLLL
  >>> from fpylll import Enumeration, EvaluatorStrategy
  >>> FPLLL.set_random_seed(1337)
  >>> def gauss(radius, dim, nr):
  ...  A = IntegerMatrix.identity(dim)   #define the latttice Z^dim
  ...  M = MatGSO(A)
  ...  _ = M.update_gso()
  ...  enum = Enumeration(M, nr_solutions = nr)
  ...  e1 = enum.enumerate(0, dim, radius**2, 0)
  ...  return [tuple(dim*[0])] + [v for d,v in e1] + [tuple([-x for x in v]) for d,v in e1]

For instance `N_2` is given by

::

  >>> g = gauss(2, 2, 100)
  >>> len(g)
  13
  >>> g
  [(0, 0), (0.0, 1.0), (1.0, 0.0), (-1.0, 1.0), (1.0, 1.0), (0.0, 2.0), (2.0, 0.0), (-0.0, -1.0), (-1.0, -0.0), (1.0, -1.0), (-1.0, -1.0), (-0.0, -2.0), (-2.0, -0.0)]


For `{\rm dim} = 2` is enough to choose the parameter `nr = \lceil \pi R^2+2\sqrt{2}\pi R\rceil.` For `R=80` we get

::

  >>> from math import pi, ceil, sqrt
  >>> R = 80
  >>> nr = ceil(pi*R**2 + 2*sqrt(2)*pi*R)
  >>> len(gauss(R,2,nr))
  20081

The parameter `nr_solutions` is by default `1.` If we set say, `{\rm{nr\_solutions}}= nr = \lceil \pi R^2+2\sqrt{2}\pi R\rceil,` then the enumerate function will return, say the set `\mathcal{A}_R=\{{\bf x}_1,...,{\bf x}_{n}\}` `(n<nr)` and `{\bf x}_i\not={\bf 0}.` The set `\mathcal{A}_R^-=\{-{\bf x}_1,...,-{\bf x}_{n}\}` is such that `{\mathcal{A}_R}\cap \mathcal{A}_R^{-}=\emptyset.` That is, the enumerate function returns only a set of vectors ignoring the negative ones. So the set  `\{(x,y)\in{\mathbb{Z}}^2 : x^2+y^2\leq R^2\}=\mathcal{A}_R\cup \mathcal{A}_R^{-}\cup \{ {\bf 0}\}.` Therefore `N_R = 2|\mathcal{A}_R|+1.` We can use the following code for computing `N_R`:

::

  >>> # computation of N_R
  >>> from fpylll.fplll.gso import MatGSO
  >>> from fpylll.fplll.integer_matrix import IntegerMatrix
  >>> from fpylll import Enumeration, EvaluatorStrategy
  >>> import numpy as np
  >>> def n(radius):
  ...   dim = 2
  ...   pi = np.pi
  ...   nr = np.ceil(pi*radius**2 + 2*np.sqrt(2)*pi*radius)
  ...   B = np.matlib.identity(dim)
  ...   A = npmat2fpmat(B)   #define the latttice Z^dim
  ...   M = MatGSO(A)
  ...   _ = M.update_gso()
  ...   enum = Enumeration(M,nr_solutions = nr)
  ...   e1 = enum.enumerate(0, dim, radius**2, 0)
  ...   return 2*len(e1)+1


.. [1] https://en.wikipedia.org/wiki/Gauss_circle_problem
