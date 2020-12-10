#!/bin/sh

if [[ ! -d "./third-party" ]]; then
    mkdir ./third-party
fi
cd ./third-party

# Install Hiredis
if ls ./hiredis/install/lib/libhiredis* 1>/dev/null 2>&1; then
    echo "Hiredis has already been downloaded and installed"
    export HIREDIS_INSTALL_PATH="$(pwd)/hiredis/install"
    export LD_LIBRARY_PATH="$HIREDIS_INSTALL_PATH/lib":$LD_LIBRARY_PATH
else
    if [[ ! -d "./hiredis" ]]; then
	git clone https://github.com/redis/hiredis.git hiredis --branch master --depth=1
	echo "Hiredis downloaded"
    fi
    cd hiredis
    CC=gcc CXX=g++ make PREFIX="$(pwd)/install"
    CC=gcc CXX=g++ make PREFIX="$(pwd)/install" install
    cd ../
    export HIREDIS_INSTALL_PATH="$(pwd)/hiredis/install"
    export LD_LIBRARY_PATH="$HIREDIS_INSTALL_PATH/lib":$LD_LIBRARY_PATH
    echo "Finished installing Hiredis"
fi


#Install Redis-plus-plus
if ls ./redis-plus-plus/install/lib/libredis++* 1>/dev/null 2>&1; then
    echo "Redis-plus-plus has already been downloaded and installed"
else
    if [[ ! -d "./redis-plus-plus" ]]; then
        git clone https://github.com/sewenew/redis-plus-plus.git redis-plus-plus --branch master --depth=1
        echo "Redis-plus-plus downloaded"
    fi
    cd redis-plus-plus
    ex -s -c '2i|SET_PROPERTY(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)' -c x CMakeLists.txt
    mkdir compile
    cd compile
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="${HIREDIS_INSTALL_PATH}" -DCMAKE_INSTALL_PREFIX="$(pwd)/../install" -DCMAKE_CXX_STANDARD=17 ..
    CC=gcc CXX=g++ make -j 2
    CC=gcc CXX=g++ make install
    cd ../../
    echo "Finished installing Redis-plus-plus"
fi


# Install Protobuf
if [[ -f ./protobuf/install/bin/protoc ]]; then
    echo "Protobuf has already been downloaded and installed"
else
    if [[ ! -d "./protobuf" ]]; then
	git clone https://github.com/protocolbuffers/protobuf.git protobuf
	cd protobuf
	git checkout tags/v3.11.3
	cd ..
    else
	echo "Protobuf downloaded"
    fi
    cd protobuf
    echo "Downloading Protobuf dependencies"
    git submodule update --init --recursive
    ./autogen.sh
    ./configure --prefix="$(pwd)/install"
    CC=gcc CXX=g++ make -j 8
    CC=gcc CXX=g++ make check -j 8
    CC=gcc CXX=g++ make install
    echo "Finished installing Protobuf"
    cd ../
fi


# Install Pybind11
if [[ -d "./pybind" ]]; then
    echo "PyBind11 has already been downloaded and installed"
else
	git clone https://github.com/pybind/pybind11.git pybind --depth=1
    cd pybind
    mkdir build
    cd ..
	echo "PyBind11 downloaded"
fi


cd ../
