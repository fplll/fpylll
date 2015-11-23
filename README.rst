fpyLLL
======

A Python wrapper for `fplll <https://github.com/dstehle/fplll>`.

Requirements
------------

This package relies on the following C/C++ libraries:

- `GMP <https://gmplib.org>`_ or `MPIR <http://mpir.org>`_ for arbitrary precision integer arithmetic,
- `MPFR <http://www.mpfr.org>`_ for arbitrary precision floating point arithmetic
- `QD <http://crd-legacy.lbl.gov/~dhbailey/mpdist/>`_ for double double and quad double arithmetic,
- `fpLLL <https://github.com/dstehle/fplll>`_ for pretty much everything.

It also relies on

- `Cython <http://cython.org>`_ for linking Python and C/C++.
- `py.test <http://pytest.org/latest/>`_ for testing Python

Getting Started
---------------

We recommend `virtualenv <https://virtualenv.readthedocs.org/>`_ for isolating Python build environments and `virtualenvwrapper <https://virtualenvwrapper.readthedocs.org/>`_ to manage virtual environments.

1. create a new virtualenv and activate it::

     $ virtualenv fpylll
     $ source ./fpylll/bin/activate

We indicate active virtualenvs by the prefix ``(fpylll)``.

2. install required libraries – `GMP <https://gmplib.org>`_ or `MPIR <http://mpir.org>`_, `MPFR <http://www.mpfr.org>`_, `QD <http://crd-legacy.lbl.gov/~dhbailey/mpdist/>`_ – if not available system-wide. Then, execute::

     $ (fpylll) pip install -r requirements.txt

to install `Cython <http://cython.org>`_ and `pytest <http://pytest.org/latest/>`_.

3. install the ``fpylll-changes`` branch of fplll from https://github.com/malb/fplll in this virtualenv::

     $ (fpylll) cd <path-to-your-fplll>
     $ (fpylll) ./configure –prefix=$VIRTUAL_ENV
     $ (fpylll) make install

4. build the Python extension::

     $ (fpylll) cd <path-to-the-python-extension>
     $ (fpylll) python setup.py build_ext
     $ (fpylll) python setup.py install

5. You will need to::

     $ (fpylll) export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib"

so that Python can find fplll and friends.

6. You may want to change out of the root directory of this repository before starting ``(i)python``, as the presence of a ``fpylll`` directory tends to confuse its module finding. For example::

     $ (fpylll) cd tests
     $ (fpylll) ipython


Example
-------

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

The interface already allows to implement the basic BKZ algorithm in about 60 pretty readable lines of Python code (cf. `bkz.py <https://github.com/malb/fpylll/blob/master/examples/simple_bkz.py>`_).

Implementation Stuff
--------------------

- I copied a decent bit of code over from Sage, mostly from it’s fpLLL interface. But I also imported Sage’s excellent interrupt handling routines.

- I had to make some minor changes to some C++ files, essentially inlining more functions. The trouble with templated C++ is that the compiler seem not to like to instantiate small-ish functions which are called only once, even if they are not inlined. Hence, those symbols were missing and I had to work around that.

- I stuck to fpLLL’s naming conventions in general except for a few cases where they were rather “un-Pythonic“.

- Pull requests etc. welcome.
