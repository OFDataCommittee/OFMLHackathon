#!/bin/bash


# setup the necessary environment variables for testing and builds
# this must be *sourced* in the top level smartsim directory in the
# shell that will be used for building.

echo "Setting up SILC environment"

if ls ./third-party/hiredis/install/lib/libhiredis* 1>/dev/null 2>&1; then
    export HIREDIS_INSTALL_PATH="$(pwd)/third-party/hiredis/install"
    export LD_LIBRARY_PATH="$HIREDIS_INSTALL_PATH/lib":$LD_LIBRARY_PATH
    echo "Set Hiredis install path to $HIREDIS_INSTALL_PATH"
fi

if ls ./third-party/redis-plus-plus/install/lib/libredis++* 1>/dev/null 2>&1; then
    export REDISPP_INSTALL_PATH="$(pwd)/third-party/redis-plus-plus/install"
    export LD_LIBRARY_PATH="$REDISPP_INSTALL_PATH/lib":$LD_LIBRARY_PATH
    echo "Set RedisPP install path to $REDISPP_INSTALL_PATH"
fi

if [[ -f ./third-party/protobuf/install/bin/protoc ]]; then
    export PROTOBUF_INSTALL_PATH="$(pwd)/third-party/protobuf/install"
    export LD_LIBRARY_PATH="$PROTOBUF_INSTALL_PATH/lib":$LD_LIBRARY_PATH
    export PATH="$PROTOBUF_INSTALL_PATH/bin":$PATH
    echo "Set Protobuf install path to $PROTOBUF_INSTALL_PATH"
    echo "Added protoc to PATH"
fi

if [[ -d "./third-party/pybind" ]]; then
    export PYBIND_INCLUDE_PATH="$(pwd)/third-party/pybind/include/pybind11/"
    export PYBIND_INSTALL_PATH="$(pwd)/third-party/pybind/"
    echo "Set Pybind install path to $PYBIND_INSTALL_PATH"
fi

# testings dep environment variables
# Redis
if [[ -f ./third-party/redis/src/redis-server ]]; then
    export REDIS_INSTALL_PATH="$(pwd)/third-party/redis/src"
    echo "Set Redis server install path to $REDIS_INSTALL_PATH"
fi

# detect RedisAI CPU installation
if [[ -f ./third-party/RedisAI/install-cpu/redisai.so ]]; then
    export REDISAI_CPU_INSTALL_PATH="$(pwd)/third-party/RedisAI/install-cpu"
    echo "Set RedisAI CPU install path to $REDISAI_CPU_INSTALL_PATH"
fi

# detect RedisAI GPU installation
if [[ -f ./third-party/RedisAI/install-gpu/redisai.so ]]; then
    export REDISAI_GPU_INSTALL_PATH="$(pwd)/third-party/RedisAI/install-gpu"
    echo "Set RedisAI GPU install path to $REDISAI_GPU_INSTALL_PATH"
fi

# Update PYTHONPATH
if [[ ":$PYTHONPATH:" != *"$(pwd)/src/python/module/"* ]]; then
    echo "Adding SILC to PYTHONPATH"
    export PYTHONPATH="$(pwd)/src/python/module/:${PYTHONPATH}"
    echo $PYTHONPATH
else
    echo "SILC found in PYTHONPATH"
fi

# Set SILC_INSTALL_PATH for external application build assistance
export SILC_INSTALL_PATH="$(pwd)"
echo "Setting the SILC install path to $SILC_INSTALL_PATH"