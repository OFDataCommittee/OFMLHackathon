#!/bin/bash

if [[ ! -d "./third-party" ]]; then
    mkdir ./third-party
fi

if [[ ! -d "./install" ]]; then
    mkdir ./install
fi

cd ./third-party

# get the number of processors
NPROC=$(python -c "import multiprocessing as mp; print(mp.cpu_count())")
# get python installed cmake
CMAKE=$(python -c "import cmake; import os; print(os.path.join(cmake.CMAKE_BIN_DIR, 'cmake'))")

# Install Hiredis
if ls ../install/lib/libhiredis.a 1>/dev/null 2>&1; then
    echo "Hiredis has already been installed"
else
    if [[ ! -d "./hiredis" ]]; then
	git clone https://github.com/redis/hiredis.git hiredis --branch v1.0.0 --depth=1
	echo "Hiredis downloaded"
    fi
    cd hiredis

    LIBRARY_PATH=lib CC=gcc CXX=g++ make PREFIX="$(pwd)/../../install" static -j $NPROC
    LIBRARY_PATH=lib CC=gcc CXX=g++ make PREFIX="$(pwd)/../../install" install
    cd ../
    # delete shared libraries
    rm ../install/lib/*.so
    rm ../install/lib/*.dylib
    echo "Finished installing Hiredis"
fi


#Install Redis-plus-plus
if ls ../install/lib/libredis++.a 1>/dev/null 2>&1; then
    echo "Redis-plus-plus has already been installed"
else
    if [[ ! -d "./redis-plus-plus" ]]; then
        git clone https://github.com/sewenew/redis-plus-plus.git redis-plus-plus --branch 1.2.3 --depth=1
	    echo "Redis-plus-plus downloaded"
    fi
    cd redis-plus-plus
    #ex -s -c '2i|SET_PROPERTY(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)' -c x CMakeLists.txt
    mkdir compile
    cd compile

    $CMAKE -DCMAKE_BUILD_TYPE=Release -DREDIS_PLUS_PLUS_BUILD_TEST=OFF -DREDIS_PLUS_PLUS_BUILD_SHARED=OFF -DCMAKE_PREFIX_PATH="$(pwd)../../../install/lib/" -DCMAKE_INSTALL_PREFIX="$(pwd)/../../../install" -DCMAKE_CXX_STANDARD=17 ..
    CC=gcc CXX=g++ make -j $NPROC
    CC=gcc CXX=g++ make install
    cd ../../
    echo "Finished installing Redis-plus-plus"
fi

# Install Pybind11
if [[ -d "./pybind" ]]; then
    echo "PyBind11 has already been downloaded and installed"
else
    git clone https://github.com/pybind/pybind11.git pybind --branch v2.6.2 --depth=1
    cd pybind
    mkdir build
    cd ..
    echo "PyBind11 downloaded"
fi


cd ../
