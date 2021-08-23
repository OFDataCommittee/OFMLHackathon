#!/bin/bash

# get the number of processors
NPROC=$(python -c "import multiprocessing as mp; print(mp.cpu_count())")

#Install RedisAI
if [[ -f ./RedisAI/install-cpu/redisai.so ]]; then
    echo "RedisAI CPU has already been downloaded and installed"
else
    if [[ ! -d "./RedisAI" ]]; then
        GIT_LFS_SKIP_SMUDGE=1 git clone --recursive https://github.com/RedisAI/RedisAI.git --branch v1.2.3 --depth=1 RedisAI
        cd RedisAI
        cd ..
    else
        echo "RedisAI downloaded"
    fi
    cd RedisAI
    echo "Downloading RedisAI CPU dependencies"
    CC=gcc CXX=g++ WITH_PT=1 WITH_TF=1 WITH_TFLITE=0 WITH_ORT=0 bash get_deps.sh cpu
    echo "Building RedisAI"
    CC=gcc CXX=g++ GPU=0 WITH_PT=1 WITH_TF=1 WITH_TFLITE=0 WITH_ORT=0 WITH_UNIT_TESTS=0 make -j $NPROC -C opt clean build

    if [ -f "./install-cpu/redisai.so" ]; then
        echo "Finished installing RedisAI"
        cd ../
    else
        echo "ERROR: RedisAI failed to build"
        exit 1
    fi
fi
