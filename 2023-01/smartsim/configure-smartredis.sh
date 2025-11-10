#!/usr/bin/env bash

# Define environmental variables for including and linking SmartRedis in
# OpenFOAM applications and libraries.

export FOAM_SMARTREDIS=$PWD/smartredis
if [ ! -d "$FOAM_SMARTREDIS" ]; then
    echo "$FOAM_SMARTREDIS does not exist, please source configure-smartredis.sh from its folder"
fi
export SMARTREDIS_INCLUDE=$FOAM_SMARTREDIS/install/include
export SMARTREDIS_LIB=$FOAM_SMARTREDIS/install/lib
export LD_LIBRARY_PATH=$FOAM_SMARTREDIS_LIB:$LD_LIBRARY_PATH
export SSDB="127.0.0.1:8000" # for multinode setup let smartsim do this
