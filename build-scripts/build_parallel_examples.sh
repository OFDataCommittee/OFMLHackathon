#!/bin/bash

cd ./examples/parallel/fortran/

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




DO_FORTRAN="yes"

if [ "$(uname)" == "Darwin" ]; then
    DO_FORTRAN="no"
fi

if [[ $DO_FORTRAN == "yes" ]]; then

    # setup build dirs
    mkdir build
    cd ./build
  
    # TODO add platform dependent build step here
    cmake ..

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

    cd ../../cpp

    echo "Fortran parallel examples built"
else
    echo "Skipping Fortran parallel example build"
fi



# setup build dirs
mkdir build
cd ./build

# TODO add platform dependent build step here
cmake ..

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
