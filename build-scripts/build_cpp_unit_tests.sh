#!/bin/bash

# get the number of processors
NPROC=$(python -c "import multiprocessing as mp; print(mp.cpu_count())")

CMAKE=$(python -c "import cmake; import os; print(os.path.join(cmake.CMAKE_BIN_DIR, 'cmake'))")

BASEDIR=$(pwd)

cd ./tests/cpp/unit-tests

if [[ -d "./build" ]]; then
    echo "Removing previous cpp unit test build directory"
    rm -rf ./build
fi

# setup build dirs
mkdir build
cd ./build

# TODO add platform dependent build step here
$CMAKE ..

if [ $? != 0 ]; then
    echo "ERROR: cmake for CPP tests failed"
    cd ..
    exit 1
fi

make -j$NPROC

if [ $? != 0 ]; then
    echo "ERROR: failed to make CPP tests"
    cd ..
    exit 1
fi

cd $BASEDIR