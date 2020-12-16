#!/bin/bash

cd ./tests/fortran/

if [ -z "$HIREDIS_INSTALL_PATH" ]; then
    echo "WARNING: HIREDIS_INSTALL_PATH is not set"
    echo "Test may fail to build"
else
    echo "Found HIREDIS_INSTALL_PATH: $HIREDIS_INSTALL_PATH"
fi

if [ -z "$REDISPP_INSTALL_PATH" ]; then
    echo "WARNING: REDISPP_INSTALL_PATH is not set"
    echo "Tests may fail to build"
else
    echo "Found REDISPP_INSTALL_PATH: $REDISPP_INSTALL_PATH"
fi

if [ -z "$PROTOBUF_INSTALL_PATH" ]; then
    echo "WARNING: PROTOBUF_INSTALL_PATH is not set"
    echo "Tests may fail to build"
else
    echo "Found PROTOBUF_INSTALL_PATH: $PROTOBUF_INSTALL_PATH"
fi



# setup build dirs
mkdir build
cd ./build

DO_FORTRAN="yes"

if [ "$(uname)" == "Darwin" ]; then
    DO_FORTRAN="no"
fi

if [[ $DO_FORTRAN == "yes" ]]; then
    # TODO add platform dependent build step here
    cmake ..

    if [ $? != 0 ]; then
        echo "ERROR: cmake for Fortran tests failed"
        cd ..
        exit 1
    fi

    make -j 4

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

