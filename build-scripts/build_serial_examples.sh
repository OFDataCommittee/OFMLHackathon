#!/bin/bash

CMAKE=$(python -c "import cmake; import os; print(os.path.join(cmake.CMAKE_BIN_DIR, 'cmake'))")

cd ./examples/serial/c/

# setup build dirs
mkdir build
cd ./build

# TODO add platform dependent build step here
$CMAKE ..

if [ $? != 0 ]; then
    echo "ERROR: cmake for C serial examples failed"
    cd ..
    exit 1
fi

make -j 4

if [ $? != 0 ]; then
    echo "ERROR: failed to make C serial examples"
    cd ..
    exit 1
fi

cd ../

echo

cd ../fortran


DO_FORTRAN="yes"

if [ "$(uname)" == "Darwin" ]; then
    DO_FORTRAN="yes"
fi

if [[ $DO_FORTRAN == "yes" ]]; then

    # setup build dirs
    mkdir build
    cd ./build

    # TODO add platform dependent build step here
    $CMAKE ..

    if [ $? != 0 ]; then
        echo "ERROR: cmake for parallel Fortran examples failed"
        cd ..
        exit 1
    fi

    make

    if [ $? != 0 ]; then
        echo "ERROR: failed to make Fortran parallel examples"
        cd ..
        exit 1
    fi

    cd ../

    echo "Fortran parallel examples built"
else
    echo "Skipping Fortran parallel example build"
fi

cd ../cpp/

# setup build dirs
mkdir build
cd ./build

# TODO add platform dependent build step here
$CMAKE ..

if [ $? != 0 ]; then
    echo "ERROR: cmake for CPP parallel examples failed"
    cd ..
    exit 1
fi

make -j 4

if [ $? != 0 ]; then
    echo "ERROR: failed to make CPP parallel examples"
    cd ..
    exit 1
fi
