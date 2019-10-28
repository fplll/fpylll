#!/usr/bin/env bash

# Create Virtual Environment

rm -rf fpylll-env
virtualenv fpylll-env
cat <<EOF >>fpylll-env/bin/activate
### LD_LIBRARY_HACK
_OLD_LD_LIBRARY_PATH="\$LD_LIBRARY_PATH"
LD_LIBRARY_PATH="\$VIRTUAL_ENV/lib:\$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH
### END_LD_LIBRARY_HACK

### PKG_CONFIG_HACK
_OLD_PKG_CONFIG_PATH="\$PKG_CONFIG_PATH"
PKG_CONFIG_PATH="\$VIRTUAL_ENV/lib/pkgconfig:\$PKG_CONFIG_PATH"
export PKG_CONFIG_PATH
### END_PKG_CONFIG_HACK
      
CFLAGS="\$CFLAGS -O3 -march=native -Wp,-U_FORTIFY_SOURCE"
CXXFLAGS="\$CXXFLAGS -O3 -march=native -Wp,-U_FORTIFY_SOURCE"
export CFLAGS
export CXXFLAGS
EOF

ln -s fpylll-env/bin/activate

source activate

pip install -U pip

# Install FPLLL

git clone https://github.com/fplll/fplll fpylll-fplll
cd fpylll-fplll || exit
./autogen.sh
./configure --prefix="$VIRTUAL_ENV" $CONFIGURE_FLAGS
make clean
make -j 4
make install
cd ..

# Install FPyLLL
pip install Cython
pip install -r requirements.txt
pip install -r suggestions.txt
python setup.py clean
python setup.py build_ext
python setup.py install


echo "Don't forget to activate environment:"
echo " source activate"
