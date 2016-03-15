.. image:: https://travis-ci.org/malb/fpylll.svg
    :target: https://travis-ci.org/malb/fpylll

A Python wrapper for `fplll <https://github.com/dstehle/fplll>`_.

Requirements
------------

This package relies on the following C/C++ libraries:

- `GMP <https://gmplib.org>`_ or `MPIR <http://mpir.org>`_ for arbitrary precision integer arithmetic,
- `MPFR <http://www.mpfr.org>`_ for arbitrary precision floating point arithmetic
- `QD <http://crd-legacy.lbl.gov/~dhbailey/mpdist/>`_ for double double and quad double arithmetic,
- `fpLLL <https://github.com/dstehle/fplll>`_ for pretty much everything.

It also relies on

- `Cython <http://cython.org>`_ for linking Python and C/C++.
- `cysignals <https://github.com/sagemath/cysignals>`_ for signal handling such as interrupting C++ code
- `py.test <http://pytest.org/latest/>`_ for testing Python

We also suggest

- `IPython  <https://ipython.org>`_ for interacting with Python
- `Numpy <http://www.numpy.org>`_ for numerical computations (e.g. with Gram-Schmidt values)

Getting Started
---------------

We recommend `virtualenv <https://virtualenv.readthedocs.org/>`_ for isolating Python build environments and `virtualenvwrapper <https://virtualenvwrapper.readthedocs.org/>`_ to manage virtual environments.

1. Create a new virtualenv and activate it::

     $ virtualenv env
     $ source ./env/bin/activate

We indicate active virtualenvs by the prefix ``(fpylll)``.

2. Install the required libraries – `GMP <https://gmplib.org>`_ or `MPIR <http://mpir.org>`_, `MPFR <http://www.mpfr.org>`_ and `QD <http://crd-legacy.lbl.gov/~dhbailey/mpdist/>`_ – if not available already.

3. Install the ``fpylll-changes`` branch of fplll from https://github.com/malb/fplll to the virtual environment::

     $ (env) ./install-dependencies.sh $VIRTUAL_ENV

4. Then, execute::

     $ (env) pip install Cython
     $ (env) pip install -r requirements.txt

to install the required Python packages (see above).

5. If you are so inclined, run::

     $ (env) pip install -r suggestions.txt

to install suggested Python packages as well (optional).

6. Build the Python extension::

     $ (env) python setup.py build_ext
     $ (env) python setup.py install

7. To run fpylll, you will need to::

     $ (env) export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib"

so that Python can find fplll and friends.

8. Start Python::

    $ (env) ipython

To reactivate the virtual environment later, simply run::

    $ source ./env/bin/activate

Note that you can also patch ``activate`` to set ``LD_LIBRRY_PATH``. For this, add::

    ### LD_LIBRARY_HACK
    _OLD_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
    LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH"
    export LD_LIBRARY_PATH
    ### END_LD_LIBRARY_HACK

towards the end and::

    ### LD_LIBRARY_HACK
    if ! [ -z ${_OLD_LD_LIBRARY_PATH+x} ] ; then
        LD_LIBRARY_PATH="$_OLD_LD_LIBRARY_PATH"
        export LD_LIBRARY_PATH
        unset _OLD_LD_LIBRARY_PATH
    fi
    ### END_LD_LIBRARY_HACK

in the ``deactivate`` function in the ``activate`` script.

Example
-------

Here is an example session:

    >>> from fpylll import *

    >>> A = IntegerMatrix(50, 50)
    >>> A.randomize("ntrulike", bits=50, q=127)
    >>> A[0].norm()
    3564748886669202.5
    >>> LLL.reduction(A)
    >>> A[0].norm()
    24.06241883103193

    >>> A = IntegerMatrix(50, 50)
    >>> A.randomize("ntrulike", bits=50, q=127)
    >>> M = GSO.Mat(A)
    >>> M.update_gso()
    >>> M.get_mu(1,0)
    0.815748944429783

    >>> L = LLL.Reduction(M)
    >>> L()
    >>> M.get_mu(1,0)
    0.41812865497076024

The interface already allows to implement the basic BKZ algorithm in about 60 pretty readable lines of Python code (cf. `bkz.py <https://github.com/malb/fpylll/blob/master/src/fpylll/contrib/simple_bkz.py>`_).

Implementation Stuff
--------------------

- We copied a decent bit of code over from Sage, mostly from it’s fpLLL interface.

- We stuck to fpLLL’s naming conventions in general except for a few cases where they were rather “un-Pythonic“.

- Pull requests etc. welcome.

- We run `flake8 <https://flake8.readthedocs.org/en/latest/>`_ on every commit automatically, In particular, we run::

    flake8 --max-line-length=120 --max-complexity=16 --ignore=E22,E241 src

  See `.travis.yml <https://github.com/malb/fpylll/blob/master/.travis.yml>`_ for details.
