#!/bin/bash

# get the number of processors
NPROC=$(python -c "import multiprocessing as mp; print(mp.cpu_count())")

#Install RedisAI
if [[ -f ./RedisAI/install-gpu/redisai.so ]]; then
    echo "RedisAI GPU has already been downloaded and installed"
else

    # check for cudnn includes
    if [ -z "$CUDA_HOME" ]; then
        echo "ERROR: CUDA_HOME is not set"
        exit 1
    else
        echo "Found CUDA_HOME: $CUDA_HOME"
    fi

    # check for cudnn includes
    if [ -z "$CUDNN_INCLUDE_DIR" ]; then
        echo "ERROR: CUDNN_INCLUDE_DIR is not set"
        exit 1
    else
        echo "Found CUDNN_INCLUDE_DIR: $CUDNN_INCLUDE_DIR "
        if [ -f "$CUDNN_INCLUDE_DIR/cudnn.h" ]; then
            echo "Found cudnn.h at $CUDNN_INCLUDE_DIR"
        else
            echo "ERROR: could not find cudnn.h at $CUDNN_INCLUDE_DIR"
            exit 1
        fi
    fi

    # check for cudnn library
    if [ -z "$CUDNN_LIBRARY" ]; then
        echo "ERROR: CUDNN_LIBRARY is not set"
        exit 1
    else
        echo "Found CUDNN_LIBRARY: $CUDNN_LIBRARY"
        if [ -f "$CUDNN_LIBRARY/libcudnn.so" ]; then
            echo "Found libcudnn.so at $CUDNN_LIBRARY"
        else
            echo "ERROR: could not find libcudnn.so at $CUDNN_LIBRARY"
            exit 1
        fi
    fi


    if [[ ! -d "./RedisAI" ]]; then
        GIT_LFS_SKIP_SMUDGE=1 git clone --recursive https://github.com/RedisAI/RedisAI.git --branch v1.2.3 --depth=1 RedisAI
        cd RedisAI
        cd ..
    else
        echo "RedisAI downloaded"
    fi
    cd RedisAI
    echo "Downloading RedisAI CPU dependencies"
    CC=gcc CXX=g++ WITH_PT=1 WITH_TF=1 WITH_TFLITE=0 WITH_ORT=0 bash get_deps.sh gpu
    echo "Building RedisAI"
    CC=gcc CXX=g++ GPU=1 WITH_PT=1 WITH_TF=1 WITH_TFLITE=0 WITH_ORT=0 WITH_UNIT_TESTS=0 make -j $NPROC -C opt clean build

    if [ -f "./install-gpu/redisai.so" ]; then
        echo "Finished installing RedisAI"
        cd ../
    else
        echo "ERROR: RedisAI failed to build"
        exit 1
    fi
fi
