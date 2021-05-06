#!/bin/bash


# setup the necessary environment variables for testing and builds
# this must be *sourced* in the top level smartsim directory in the
# shell that will be used for building.

echo "Setting up SmartRedis environment for testing"

export SMARTREDIS_TEST_CLUSTER=True
echo SMARTREDIS_TEST_CLUSTER set to $SMARTREDIS_TEST_CLUSTER

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
