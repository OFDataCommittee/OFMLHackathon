#!/bin/bash

CMAKE=$(which cmake)

cd ./tests/c/

# setup build dirs
mkdir build
cd ./build

# TODO add platform dependent build step here
$CMAKE ..

if [ $? != 0 ]; then
    echo "ERROR: cmake for C tests failed"
    cd ..
    exit 1
fi

make -j 4

if [ $? != 0 ]; then
    echo "ERROR: failed to make C tests"
    cd ..
    exit 1
fi

cd ../

echo