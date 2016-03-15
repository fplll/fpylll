#!/bin/bash
git clone https://github.com/dstehle/fplll

cd fplll
./autogen.sh
if test "$1" != ""; then
    ./configure --prefix="$1"
else
    ./configure
fi

make
make install

cd ..
rm -rf fplll
