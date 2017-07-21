.. role:: math(raw)
   :format: html latex
..

.. role:: raw-latex(raw)
   :format: latex
..

Linear Pruning
==============

If we want to use pruning we can use the default pruning of fplll or
to use our custom pruning. For instance, say that for some reason we
want to use linear pruning. Then, we shall define the new linear pruning
strategy as follows.

::

  >>> def linear_pruning_strategy(blocksize,level):
  ...  if level > blocksize - 1:
  ...    print "error"
  ...    return
  ...  if blocksize  < 5:
  ...    print "error"
  ...    return
  ...  from fpylll import BKZ
  ...  from fpylll.fplll.bkz_param import Pruning
  ...  from fpylll.fplll.bkz_param import Strategy
  ...  preprocessing = 3
  ...  strategies1 = [Strategy(i) for i in range(6)]
  ...  for b in range(6, blocksize+1):
  ...    pr  = Pruning.LinearPruning(blocksize, level)
  ...    s   = Strategy(b, [preprocessing], [pr] )
  ...    strategies1.append(s)
  ...  param = BKZ.Param( block_size = blocksize, strategies = strategies1 )
  ...  return param

So, now we can define a new strategy that uses linear pruning

::

  >>> LP = linear_pruning_strategy(10, 6)

Now, we shall compute the BKZ reduction of a large matrix using linear
pruning.

::
  >>> from fpylll import IntegerMatrix, BKZ, set_random_seed
  >>> A = IntegerMatrix(140, 141)
  >>> set_random_seed(2013)
  >>> A.randomize("intrel", bits=100)
  >>> bkz_reduced = BKZ.reduction(A, LP)

Now, `bkz_reduced` is the BKZ reduced matrix of **A** using linear
pruning with blocksize 10 and level 6. If we want to use the default
strategy of fplll (which is faster than the previous linear pruning
strategy) we use `BKZ.DEFAULT_STRATEGY`,

::

  >>> param = BKZ.Param(block_size=10, strategies=BKZ.DEFAULT_STRATEGY)
  >>> bkz_reduced_2 = BKZ.reduction(A, param)

and

::

  >>> bkz_reduced == bkz_reduced_2
  True
