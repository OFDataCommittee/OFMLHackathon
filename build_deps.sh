#!/bin/sh

if [[ ! -d "./third-party" ]]; then
    mkdir third-party
fi
cd third-party

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

# Install Keydb
found_keydb=$(which keydb-server > /dev/null 2<&1)
if [[ -x "$found_keydb" ]] ; then
    echo "KeyDB is installed"
else
    if [[ -d "./KeyDB" ]] ; then
        echo "KeyDB has already been downloaded"
        export PATH="$(pwd)/KeyDB/src:${PATH}"
        echo "Added KeyDB to PATH"
    else
        echo "Installing KeyDB"
        git clone https://github.com/JohnSully/KeyDB.git --branch v6.0.13 --depth=1
	#git clone https://github.com/JohnSully/KeyDB.git --depth=1
        cd KeyDB/
	CC=gcc CXX=g++ make -j 2
        cd ..
        export PATH="$(pwd)/KeyDB/src:${PATH}"
        echo "Finished installing KeyDB"
    fi
fi

#Install Redis-plus-plus
if ls ./redis-plus-plus/install/lib/libredis++* 1>/dev/null 2>&1; then
    echo "Redis-plus-plus has already been downloaded and installed"
    export REDISPP_INSTALL_PATH="$(pwd)/redis-plus-plus/install"
    export LD_LIBRARY_PATH="$REDISPP_INSTALL_PATH/lib":$LD_LIBRARY_PATH
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
    export REDISPP_INSTALL_PATH="$(pwd)/redis-plus-plus/install"
    export LD_LIBRARY_PATH="$REDISPP_INSTALL_PATH/lib":$LD_LIBRARY_PATH
    echo "Finished installing Redis-plus-plus"
fi

# Install Protobuf
if [[ -f ./protobuf/install/bin/protoc ]]; then
    echo "Protobuf has already been downloaded and installed"
    export PROTOBUF_INSTALL_PATH="$(pwd)/protobuf/install"
    export LD_LIBRARY_PATH="$PROTOBUF_INSTALL_PATH/lib":$LD_LIBRARY_PATH
    export PATH="$PROTOBUF_INSTALL_PATH/bin":$PATH
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
    export PROTOBUF_INSTALL_PATH="$(pwd)/install"
    export LD_LIBRARY_PATH="$PROTOBUF_INSTALL_PATH/lib":$LD_LIBRARY_PATH
    export PATH="$PROTOBUF_INSTALL_PATH/bin":$PATH
    echo "Finished installing Protobuf"
    cd ../
fi

#Install Redis
if [[ -f ./redis/src/redis-server ]]; then
    echo "Redis has already been downloaded and installed"
    export REDIS_INSTALL_PATH="$(pwd)/redis/src"
else
    if [[ ! -d "./redis" ]]; then
	git clone https://github.com/redis/redis.git redis
	cd redis
	git checkout tags/v6.0.8
	cd ..
    else
	echo "Redis downloaded"
    fi
    cd redis
    echo "Downloading redis dependencies"
    CC=gcc CXX=g++ make
    export REDIS_INSTALL_PATH="$(pwd)/src"
    echo "Finished installing redis"
    cd ../
fi

#Install RedisAI CPU
if [[ -f ./RedisAI/install-cpu/redisai.so ]]; then
    echo "RedisAI CPU has already been downloaded and installed"
    export REDISAI_CPU_INSTALL_PATH="$(pwd)/RedisAI/install-cpu"
else
    if [[ ! -d "./RedisAI" ]]; then
	git clone https://github.com/RedisAI/RedisAI.git RedisAI
	cd RedisAI
	git checkout tags/v1.0.2
	cd ..
    else
	echo "RedisAI downloaded"
    fi
    cd RedisAI
    echo "Downloading RedisAI dependencies"
    CC=gcc CXX=g++ bash get_deps.sh cpu
    CC=gcc CXX=g++ ALL=1 make -C opt clean build
    export REDISAI_CPU_INSTALL_PATH="$(pwd)/install-cpu"
    echo "Finished installing RedisAI"
    cd ../
fi

# Install Pybind11
if [[ -d "./pybind" ]]; then
    echo "PyBind11 has already been downloaded and installed"
    export PYBIND_INCLUDE_PATH="$(pwd)/pybind/include/pybind11/"
    export PYBIND_INSTALL_PATH="$(pwd)/pybind/"
else
	git clone https://github.com/pybind/pybind11.git pybind --depth=1
    cd pybind
    mkdir build
    cd ..
	echo "PyBind11 downloaded"
fi


cd ../
