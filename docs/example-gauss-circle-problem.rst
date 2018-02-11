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
  >>> import numpy as np
  >>> FPLLL.set_random_seed(1337)
  >>> from numpy import matrix,matlib
  >>> def gauss(radius, dim, nr):
  ...  A = IntegerMatrix.identity(dim)   #define the latttice Z^dim
  ...  M = MatGSO(A)
  ...  _ = M.update_gso()
  ...  enum = Enumeration(M, nr_solutions = nr)
  ...  e1 = enum.enumerate(0, dim, radius**2, 0)
  ...  e2 = [np.matrix(dim*[0])] + [np.matrix(item[0]) for item in e1]
  ...  e3 = [((-1)*np.matrix(item[0])) for item in e1]
  ...  return e2+e3

For instance `N_2` is given by

::

  >>> g = gauss(2,2,100)
  >>> len(g)
  13
  >>> g
  [matrix([[0, 0]]), matrix([[ 1.]]), matrix([[ 1.]]), matrix([[ 2.]]), matrix([[ 2.]]), matrix([[ 4.]]), matrix([[ 4.]]), matrix([[-1.]]), matrix([[-1.]]), matrix([[-2.]]), matrix([[-2.]]), matrix([[-4.]]), matrix([[-4.]])]


For `{\rm dim} = 2` is enough to choose the parameter `nr = \lceil \pi R^2+2\sqrt{2}\pi R\rceil.` For `R=80` we get

::

  >>> pi = np.pi
  >>> R = 80
  >>> nr = np.ceil(pi*R**2 + 2*np.sqrt(2)*pi*R)
  >>> len(gauss(R,2,nr))
  20081

The parameter `nr_solutions` is by default `1.` If we set say, `{\rm{nr\_solutions}}= nr = \lceil \pi R^2+2\sqrt{2}\pi R\rceil,` then the enumerate function will return, say the set `\mathcal{A}_R=\{{\bf x}_1,...,{\bf x}_{n}\}` `(n<nr)` and `{\bf x}_i\not={\bf 0}.` The set `\mathcal{A}_R^-=\{-{\bf x}_1,...,-{\bf x}_{n}\}` is such that `{\mathcal{A}_R}\cap \mathcal{A}_R^{-}=\emptyset.` That is, the enumerate function returns only a set of vectors ignoring the negative ones. So the set  `\{(x,y)\in{\mathbb{Z}}^2 : x^2+y^2\leq R^2\}=\mathcal{A}_R\cup \mathcal{A}_R^{-}\cup \{ {\bf 0}\}.` Therefore `N_R = 2|\mathcal{A}_R|+1.` We can use the following code for computing `N_R`:

::

  >>> # computation of N_R
  >>> from fpylll.fplll.gso import MatGSO
  >>> from fpylll.fplll.integer_matrix import IntegerMatrix
  >>> from fpylll import Enumeration, EvaluatorStrategy
  >>> import numpy as np
  >>> from numpy import matrix,matlib
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
