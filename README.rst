fpylll
======

A Python (2 and 3) wrapper for `fplll <https://github.com/fplll/fplll>`__.

.. image:: https://travis-ci.org/fplll/fpylll.svg?branch=master
    :target: https://travis-ci.org/fplll/fpylll
.. image:: https://badge.fury.io/py/fpylll.svg
    :target: https://badge.fury.io/py/fpylll
.. image:: https://readthedocs.org/projects/fpylll/badge/?version=latest
    :target: http://fpylll.readthedocs.io/en/latest/?badge=latest

.. code-block:: python

    >>> from fpylll import *
   
    >>> A = IntegerMatrix(50, 50)
    >>> A.randomize("ntrulike", bits=50, q=127)
    >>> A[0].norm()
    3564748886669202.5

    >>> M = GSO.Mat(A)
    >>> M.update_gso()
    >>> M.get_mu(1,0)
    0.815748944429783

    >>> L = LLL.Reduction(M)
    >>> L()
    >>> M.get_mu(1,0)
    0.41812865497076024
    >>> A[0].norm()
    24.06241883103193

The basic BKZ algorithm can be implemented in about 60 pretty readable lines of Python code (cf. `simple_bkz.py <https://github.com/fplll/fpylll/blob/master/src/fpylll/algorithms/simple_bkz.py>`__).
For a quick tour of the library, you can check out the `tutorial`.
             
Requirements
------------

**fpylll** relies on the following C/C++ libraries:

- `GMP <https://gmplib.org>`__ or `MPIR <http://mpir.org>`__ for arbitrary precision integer arithmetic.
- `MPFR <http://www.mpfr.org>`__ for arbitrary precision floating point arithmetic.
- `QD <http://crd-legacy.lbl.gov/~dhbailey/mpdist/>`__ for double double and quad double arithmetic (optional).
- `fplll <https://github.com/fplll/fplll>`__ for pretty much everything.

**fpylll** also relies on

- `Cython <http://cython.org>`__ for linking Python and C/C++.
- `cysignals <https://github.com/sagemath/cysignals>`__ for signal handling such as interrupting C++ code.
- `py.test <http://pytest.org/latest/>`__ for testing Python.
- `flake8 <https://flake8.readthedocs.org/en/latest/>`__ for linting.

We also suggest

- `IPython  <https://ipython.org>`__ for interacting with Python
- `Numpy <http://www.numpy.org>`__ for numerical computations (e.g. with Gram-Schmidt values)

Online
------

**fpylll** ships with Sage 7.4. Thus, it is available via `SageMathCell <http://sagecell.sagemath.org/?z=eJxtjk1rwzAMhu-F_gfRUzpCKGODXXxwWTfGWlrWDPZBMWrjFK-2lcketPv1U0657CJePUiP1DIFaLuL9x5c6IgzXI1HGhQ8xWyPlleY2Z0rxthQKO5mJUy-kS-TEoLqu5O6kbp3OUmYjkcdu5hBf852VSQOhaCUGcXlbBKtJ2zMQMxXoljMnz-q-8WDfl3WZlu_6Hrx-C6LPWbb_ByykyFdQg82yBiKvafDyST3a9W13B-EaojyIp6NJ-qSui2h9XhMqles9JtZrteb7fT_h_8AredZkw==&lang=sage>`__ and `SageMathCloud <https://cloud.sagemath.com>`__ (select a Jupyter notebook with a Sage 7.4 kernel, the default Sage worksheet still runs Sage 7.3 at the time of writing). You can also fire up a `dply.co virtual server <https://dply.co/b/pBZ2QbxW>`__ with the latest fpylll/fplll preinstalled (it takes perhaps 15 minutes until everything is compiled).

Getting Started
---------------

**Note:** fpylll is also available via `PyPI <https://pypi.python.org/pypi/fpylll/>`__ and `Conda-Forge <https://conda-forge.github.io>`__ for `Conda <https://conda.io/docs/>`__. In what follows, we explain manual installation.

We recommend `virtualenv <https://virtualenv.readthedocs.org/>`__ for isolating Python build environments and `virtualenvwrapper <https://virtualenvwrapper.readthedocs.org/>`__ to manage virtual environments.

1. Create a new virtualenv and activate it:

   .. code-block:: bash

     $ virtualenv env
     $ source ./env/bin/activate

   We indicate active virtualenvs by the prefix ``(fpylll)``.

2. Install the required libraries - `GMP <https://gmplib.org>`__ or `MPIR <http://mpir.org>`__ and `MPFR <http://www.mpfr.org>`__  - if not available already. You may also want to install `QD <http://crd-legacy.lbl.gov/~dhbailey/mpdist/>`__.

3. Install fplll:

   .. code-block:: bash

     $ (fpylll) ./install-dependencies.sh $VIRTUAL_ENV

   Some OSX users report that they required ``export CXXFLAGS="-stdlib=lic++ -mmacosx-version-min=10.7`` and ``export CXX=clang++`` (after installing a recent clang with `brew <https://brew.sh>`__) since the default GCC installed by Apple does not have full C++11 support.
    
4. Then, execute:

   .. code-block:: bash

     $ (fpylll) pip install Cython
     $ (fpylll) pip install -r requirements.txt

   to install the required Python packages (see above).

5. If you are so inclined, run:

   .. code-block:: bash

     $ (fpylll) pip install -r suggestions.txt

   to install suggested Python packages as well (optional).

6. Build the Python extension:

   .. code-block:: bash

     $ (fpylll) export PKG_CONFIG_PATH="$VIRTUAL_ENV/lib/pkgconfig:$PKG_CONFIG_PATH"
     $ (fpylll) python setup.py build_ext
     $ (fpylll) python setup.py install

7. To run **fpylll**, you will need to:

   .. code-block:: bash

     $ (fpylll) export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib"

   so that Python can find fplll and friends.

8. Start Python:

   .. code-block:: bash

    $ (fpylll) ipython

To reactivate the virtual environment later, simply run:

   .. code-block:: bash

    $ source ./env/bin/activate

Note that you can also patch ``activate`` to set ``LD_LIBRRY_PATH``. For this, add:

.. code-block:: bash

    ### LD_LIBRARY_HACK
    _OLD_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
    LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH"
    export LD_LIBRARY_PATH
    ### END_LD_LIBRARY_HACK

    ### PKG_CONFIG_HACK
    _OLD_PKG_CONFIG_PATH="$PKG_CONFIG_PATH"
    PKG_CONFIG_PATH="$VIRTUAL_ENV/lib/pkgconfig:$PKG_CONFIG_PATH"
    export PKG_CONFIG_PATH
    ### END_PKG_CONFIG_HACK

towards the end and:

.. code-block:: bash

    ### LD_LIBRARY_HACK
    if ! [ -z ${_OLD_LD_LIBRARY_PATH+x} ] ; then
        LD_LIBRARY_PATH="$_OLD_LD_LIBRARY_PATH"
        export LD_LIBRARY_PATH
        unset _OLD_LD_LIBRARY_PATH
    fi
    ### END_LD_LIBRARY_HACK

    ### PKG_CONFIG_HACK
    if ! [ -z ${_OLD_PKG_CONFIG_PATH+x} ] ; then
        PKG_CONFIG_PATH="$_OLD_PKG_CONFIG_PATH"
        export PKG_CONFIG_PATH
        unset _OLD_PKG_CONFIG_PATH
    fi
    ### END_PKG_CONFIG_HACK

in the ``deactivate`` function in the ``activate`` script.

Multicore Support
-----------------

**fpylll** supports parallelisation on multiple cores. For all C++ support to drop the `GIL <https://wiki.python.org/moin/GlobalInterpreterLock>`_ is enabled, allowing the use of threads to parallelise. Fplll is thread safe as long as each thread works on a separate object such as ``IntegerMatrix`` or ``MatGSO``. Also, **fpylll** does not actually drop the GIL in all calls to C++ functions yet. In many scenarios using `multiprocessing <https://docs.python.org/2/library/multiprocessing.html>`_, which sidesteps the GIL and thread safety issues by using processes instead of threads, will be the better choice.

The example below calls ``LLL.reduction`` on 128 matrices of dimension 30 on four worker processes.

.. code-block:: python

    from fpylll import IntegerMatrix, LLL
    from multiprocessing import Pool

    d, workers, tasks = 30, 4, 128
    
    def run_it(p, f, A, prefix=""):
        """Print status during parallel execution."""         
        import sys
        r = []
        for i, retval in enumerate(p.imap_unordered(f, A, 1)):
            r.append(retval)
            sys.stderr.write('\r{0} done: {1:.2%}'.format(prefix, float(i)/len(A)))
            sys.stderr.flush()
        sys.stderr.write('\r{0} done {1:.2%}\n'.format(prefix, float(i+1)/len(A)))
        return r
        
    A = [IntegerMatrix.random(d, "uniform", bits=30) for _ in range(tasks)]    
    A = run_it(Pool(workers), LLL.reduction, A)

To test threading simply replace the line ``from multiprocessing import Pool`` with ``from multiprocessing.pool import ThreadPool as Pool``. For calling ``BKZ.reduction`` this way, which expects a second parameter with options, using `functools.partial <https://docs.python.org/2/library/functools.html#functools.partial>`_ is a good choice. 
    
Contributing
------------

**fpylll** welcomes contributions, cf. the list of `open issues <https://github.com/fplll/fpylll/issues>`_. To contribute, clone this repository, commit your code on a separate branch and send a pull request. Please write tests for your code. You can run them by calling::

    $ (fpylll) py.test

from the top-level directory which runs all tests in ``tests/test_*.py``. We run `flake8 <https://flake8.readthedocs.org/en/latest/>`_ on every commit automatically, In particular, we run::

    $ (fpylll) flake8 --max-line-length=120 --max-complexity=16 --ignore=E22,E241 src

Note that **fpylll** supports Python 2 and 3. In particular, tests are run using Python 2.7 and 3.5. See `.travis.yml <https://github.com/fplll/fpylll/blob/master/.travis.yml>`_ for details on automated testing.

Attribution & License
---------------------

**fpylll** is maintained by Martin Albrecht.

The following people have contributed to **fpylll**

+ E M Bray
+ Fernando Virdia
+ Guillaume Bonnoron
+ Jeroen Demeyer
+ Jérôme Benoit
+ Konstantinos Draziotis
+ Leo Ducas
+ Martin Albrecht
+ Michael Walter
+ Omer Katz
+ Fernando Virdia

We copied a decent bit of code over from Sage, mostly from it's fpLLL interface.

**fpylll** is licensed under the GPLv2+.  
