#!/bin/bash

cd ./examples/serial/c/

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

# TODO add platform dependent build step here
cmake ..

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
