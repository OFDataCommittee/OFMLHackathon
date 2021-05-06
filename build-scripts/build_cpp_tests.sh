#!/bin/bash

CMAKE=$(python -c "import cmake; import os; print(os.path.join(cmake.CMAKE_BIN_DIR, 'cmake'))")

cd ./tests/cpp/

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

make -j 4

if [ $? != 0 ]; then
    echo "ERROR: failed to make CPP tests"
    cd ..
    exit 1
fi

cd ../

echo
