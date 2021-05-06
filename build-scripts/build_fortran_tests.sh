#!/bin/bash

CMAKE=$(python -c "import cmake; import os; print(os.path.join(cmake.CMAKE_BIN_DIR, 'cmake'))")

cd ./tests/fortran/

# setup build dirs
mkdir build
cd ./build

DO_FORTRAN="yes"

if [ "$(uname)" == "Darwin" ]; then
    DO_FORTRAN="yes"
fi

if [[ $DO_FORTRAN == "yes" ]]; then
    # TODO add platform dependent build step here
    $CMAKE ..

    if [ $? != 0 ]; then
        echo "ERROR: cmake for Fortran tests failed"
        cd ..
        exit 1
    fi

    make

    if [ $? != 0 ]; then
        echo "ERROR: failed to make Fortran tests"
        cd ..
        exit 1
    fi

    cd ../
    echo "Fortran tests built"
else
    echo "Skipping Fortran test build"
fi

