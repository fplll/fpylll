#!/bin/bash

if [ "$TRAVIS_BRANCH" != "" ]; then
    FPLLL_BRANCH=$TRAVIS_BRANCH;
    CONFIGURE_FLAGS="--disable-static --with-max-enum-dim=64 --with-max-parallel-enum-dim=64"
fi

if [ "$FPLLL_BRANCH" = "" ]; then
    FPLLL_BRANCH=master
    CONFIGURE_FLAGS="--disable-static"
fi;

cloned=$(git clone https://github.com/fplll/fplll -b "$FPLLL_BRANCH")

if [ "$cloned" != "0" ]; then
    git clone https://github.com/fplll/fplll
fi

cd fplll || exit
./autogen.sh

if [ "$1" != "" ]; then
    ./configure --prefix="$1" $CONFIGURE_FLAGS
else
    ./configure $CONFIGURE_FLAGS
fi

make
make install

cd ..
rm -rf fplll
