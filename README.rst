fpyLLL
======

Contents:

.. toctree::
   :maxdepth: 2

A Python wrapper for fplll.

Getting Started
===============

This is all a bit of a hack. We assume you’re using `virtualenv <https://virtualenv.readthedocs.org/>`_ for isolating Python build environments. You can use `virtualenvwrapper <https://virtualenvwrapper.readthedocs.org/>`_ to manage your virtual enviroments.

1. create a new virtualenv and activate it::

     $ virtualenv fpylll
     $ source ./fpylll/bin/activate

We indicate active virtualenvs by the prefix ``(fpylll)``.
    
2. install the ``fpylll-changes`` branch of fplll from https://github.com/malb/fplll in this virtualenv::

     $ (fpylll) cd <path-to-your-fplll>
     $ (fpylll) ./configure –prefix=$VIRTUAL_ENV
     $ (fpylll) make install

3. build the Python extension::
       
     $ (fpylll) cd <path-to-the-python-extension>
     $ (fpylll) python setup.py build_ext
     $ (fpylll) python setup.py install

4. You will need to::

     $ (fpylll) export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib"

so that Python can find fplll and friends.

5. Start Python::

     $ (fpylll) cd tests
     $ (fpylll) ipython

You may want to change out of the root directory of this repository before starting ``(i)python``, as the presence of a ``fpylll`` directory tends to confuse its module finding.

Example
=======

The interface still rather limited, here is an example session:

    >>> from fpylll import *
    
    >>> A = IntegerMatrix(50, 50)
    >>> A.randomize("ntrulike", bits=50, q=127)
    >>> lll_reduction(A)
    
    >>> M = MatGSO(A)
    >>> L = LLLReduction(M)
    >>> M.discover_all_rows()
    >>> M.get_r_exp(0,0)
    (6.92881365683287e-310, 0)
    
    >>> L()
    >>> M.get_r_exp(0,0)
    (782.0, 0)
    
    >>> M = MatGSO(A, flags=gso.ROW_EXPO)
    >>> L = LLLReduction(M)
    >>> M.discover_all_rows()
    >>> M.get_r_exp(0,0)
    (1.976e-321, 8)
    
    >>> L()
    >>> M.get_r_exp(0,0)
    (3.0546875, 8)

The interface already allows to implement the basic BKZ algorithm in about 60 pretty readable lines of Python code (cf. `bkz.py <https://github.com/malb/fpylll/blob/master/examples/bkz.py>`_).

Implementation Stuff
====================

- I copied a decent bit of code over from Sage, mostly from it’s fpLLL interface. But I also imported Sage’s excellent interrupt handling routines.

- I had to make some minor changes to some C++ files, essentially inlining more functions. The trouble with templated C++ is that the compiler seem not to like to instantiate small-ish functions which are called only once, even if they are not inlined. Hence, those symbols were missing and I had to work around that.

- I stuck to fpLLL’s naming conventions in general except for a few cases where they were rather “un-Pythonic“.

- Pull requests etc. welcome.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
