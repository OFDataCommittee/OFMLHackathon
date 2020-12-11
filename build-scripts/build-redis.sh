#!/bin/bash

#Install Redis
if [[ -f ./redis/src/redis-server ]]; then
    echo "Redis has already been downloaded and installed"
else
    if [[ ! -d "./redis" ]]; then
        git clone https://github.com/redis/redis.git redis
        cd redis
        git checkout tags/6.0.8
        cd ..
    else
	    echo "Redis downloaded"
    fi
    cd redis
    echo "Building redis 6.0.8"
    CC=gcc CXX=g++ make MALLOC=libc
    echo "Finished installing redis"
    cd ../
fi
