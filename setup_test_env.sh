#!/bin/bash

# BSD 2-Clause License
#
# Copyright (c) 2021-2022, Hewlett Packard Enterprise
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


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
