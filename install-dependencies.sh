#!/bin/bash

if [ "$TRAVIS_BRANCH" != "" ]; then
    FPLLL_BRANCH=$TRAVIS_BRANCH;
fi

if [ "$FPLLL_BRANCH" = "" ]; then
    FPLLL_BRANCH=master
fi;

cloned=$(git clone https://github.com/fplll/fplll -b "$FPLLL_BRANCH")

if [ $cloned -ne 0 ]; then
    git clone https://github.com/fplll/fplll
fi

cd fplll || exit
./autogen.sh

if [ "$1" != "" ]; then
    ./configure --prefix="$1"
else
    ./configure
fi

make
make install

cd ..
rm -rf fplll
