#!/usr/bin/env bash

jobs="-j 4 "
if [ "$1" = "-j" ]; then
   jobs="-j $2 "
fi

# Create Virtual Environment

if [ "$PYTHON" = "" ]; then PYTHON=python; export PYTHON; fi
PIP="$PYTHON -m pip"

PYVER=$($PYTHON --version | cut -d' ' -f2)
echo "Usage:"
echo "   ./bootstrap.sh [ -j <#jobs> ]  (uses system's python)"
echo "   PYTHON=python2 ./bootstrap.sh  (uses python2)"
echo "   PYTHON=python3 ./bootstrap.sh  (uses python3)"
echo " "
echo "Using python version: $PYVER"
echo "Using $jobs"
sleep 1

rm -rf fpylll-env
$PYTHON -m virtualenv fpylll-env

if [ ! -d fpylll-env ]; then
	echo "Failed to create virtual environment in 'fpylll-env' !"
	echo "Is '$PYTHON -m virtualenv' working?"
	echo "Try '$PYTHON -m pip install virtualenv' otherwise."
	exit 1
fi

cat <<EOF >>fpylll-env/bin/activate
### LD_LIBRARY_HACK
_OLD_LD_LIBRARY_PATH="\$LD_LIBRARY_PATH"
LD_LIBRARY_PATH="\$VIRTUAL_ENV/lib:\$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH
### END_LD_LIBRARY_HACK
EOF

ln -s fpylll-env/bin/activate
source ./activate

$PIP install -U pip
$PIP install Cython
$PIP install cysignals

# Install FPLLL

cat <<EOF >>fpylll-env/bin/activate
CFLAGS="\$CFLAGS -O3 -march=native -Wp,-U_FORTIFY_SOURCE"
CXXFLAGS="\$CXXFLAGS -O3 -march=native -Wp,-U_FORTIFY_SOURCE"
export CFLAGS
export CXXFLAGS
EOF

deactivate
source ./activate

git clone https://github.com/fplll/fplll fpylll-fplll
cd fpylll-fplll || exit
./autogen.sh
./configure --prefix="$VIRTUAL_ENV" $CONFIGURE_FLAGS
make clean
make $jobs

retval=$?
if [$retval -ne 0 ]; then
    echo "Making fplll failed."
    echo "Check the logs above - they'll contain more information."
    exit 2 # 2 is the exit value if building fplll fails as a result of make $jobs.
fi

make install

if [$retval -ne 0 ]; then
    echo "Make install failed for fplll."
    echo "Check the logs above - they'll contain more information."
    exit 3 # 3 is the exit value if installing fplll failed.
fi

cd ..

$PIP install -r requirements.txt
$PIP install -r suggestions.txt

$PYTHON setup.py clean
$PYTHON setup.py build $jobs || $PYTHON setup.py build_ext
$PYTHON setup.py install

echo "Don't forget to activate environment each time:"
echo " source ./activate"
