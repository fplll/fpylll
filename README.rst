fpyLLL
======

A Python wrapper for fplll.

Getting Started
---------------

This is all a bit of a hack. We assume you’re using `virtualenv <https://virtualenv.readthedocs.org/>`_ for isolating Python build environments. You can use `virtualenvwrapper <https://virtualenvwrapper.readthedocs.org/>`_ to manage your virtual enviroments.

1. create a new virtualenv and activate it::

     $ virtualenv fpylll
     $ source ./fpylll/bin/activate

We indicate active virtualenvs by the prefix ``(fpylll)``.
    
2. install fplll (and potentially its dependencies) in this virtualenv::

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

     $ (fpylll) ipython
