#!/usr/bin/env bash

jobs="-j 4 "
if [ "$1" = "-j" ]; then
   jobs="-j $2 "
fi

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

# Create Virtual Environment

rm -rf fpylll-env activate
$PYTHON -m virtualenv fpylll-env
if [ ! -d fpylll-env ]; then
    echo "Failed to create virtual environment in 'fpylll-env'!"
    echo "Is '$PYTHON -m virtualenv' working?"
    echo "Try '$PIP install virtualenv' otherwise."
    exit 1 # 1 is the exit value if creating virtualenv fails
fi

cat <<EOF >>fpylll-env/bin/activate

### LD_LIBRARY_HACK
_OLD_LD_LIBRARY_PATH="\$LD_LIBRARY_PATH"
LD_LIBRARY_PATH="\$VIRTUAL_ENV/lib:\$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH
### END_LD_LIBRARY_HACK

CFLAGS="\$CFLAGS -O3 -march=native -Wp,-U_FORTIFY_SOURCE"
CXXFLAGS="\$CXXFLAGS -O3 -march=native -Wp,-U_FORTIFY_SOURCE"
export CFLAGS
export CXXFLAGS
EOF

ln -s fpylll-env/bin/activate
source ./activate

$PIP install -U pip -r requirements.txt -r suggestions.txt

# Install FPLLL

git clone https://github.com/fplll/fplll fpylll-fplll
cd fpylll-fplll || exit
git pull # Update if it was checked-out before
./autogen.sh
./configure --prefix="$VIRTUAL_ENV" $CONFIGURE_FLAGS

if ! make clean; then
    echo "Make clean failed in fplll. This is usually because there was an error with either autogen.sh or configure."
    echo "Check the logs above - they'll contain more information."
    exit 2 # 2 is the exit value if building fplll fails via configure or autogen
fi

if ! make $jobs; then
    echo "Making fplll failed."
    echo "Check the logs above - they'll contain more information."
    exit 3 # 3 is the exit value if building fplll fails as a result of make $jobs.
fi

if ! make install; then
    echo "Make install failed for fplll."
    echo "Check the logs above - they'll contain more information."
    exit 4 # 4 is the exit value if installing fplll failed.
fi

cd ..

# Install FPyLLL

$PYTHON setup.py clean
if ! ( $PYTHON setup.py build $jobs || $PYTHON setup.py build_ext ); then
    echo "Failed to build FPyLLL!"
    echo "Check the logs above - they'll contain more information."
    exit 5
fi
$PIP install .

# Fin

echo " "
echo "Don't forget to activate environment each time:"
echo " source ./activate"
