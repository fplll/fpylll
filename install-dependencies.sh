#!/bin/bash
git clone -b fpylll-changes https://github.com/malb/fplll
cd fplll
./autogen.sh
if test "$1" != ""; then
    ./configure --prefix=$1
else
    ./configure
fi
make
make install
cd ..
rm -rf fplll
