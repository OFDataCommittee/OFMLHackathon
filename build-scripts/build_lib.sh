#!/bin/bash

# get the number of processors
NPROC=$(python -c "import multiprocessing as mp; print(mp.cpu_count())")

# uncomment to use python installed cmake
#CMAKE=$(python -c "import cmake; import os; print(os.path.join(cmake.CMAKE_BIN_DIR, 'cmake'))")

# Remove existing module
if [ -f ./src/python/module/smartredis/smartredisPy.*.so ]; then
    echo "Removing existing module installation"
    rm ./src/python/module/smartredis/smartredisPy.*.so
fi

if [[ -d "./build" ]]; then
    echo "Removing previous build directory"
    rm -rf ./build
fi


# make a new build directory and invoke cmake
mkdir build
cd build
#$CMAKE ..
cmake ..
make -j $NPROC

if [ -f ./libsmartredis.a ]; then
    echo "Finished building libsmartredis.a"
else
    echo "ERROR: libsmartredis.a failed to compile"
    exit 1
fi

if [ -f ./smartredisPy.*.so ]; then
    echo "Finished building smartredisPy module"

    # move python module to module directory
    cp smartredisPy.*.so ../src/python/module/smartredis/
    cd ../
else
    echo "ERROR: smartredisPy module failed to compile"
    exit 1
fi


