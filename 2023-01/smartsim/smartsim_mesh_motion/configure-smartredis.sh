#!/usr/bin/env bash

# Define environmental variables for including and linking SmartRedis in
# OpenFOAM applications and libraries.

echo INFO: make sure you just sourced configure-smartredis.sh in its folder. 

export FOAM_SMARTREDIS=$HOME/smartredis
export FOAM_SMARTREDIS_INCLUDE=$FOAM_SMARTREDIS/include
export FOAM_SMARTREDIS_DEP_INCLUDE=$FOAM_SMARTREDIS/install/include
export FOAM_SMARTREDIS_LIB=$FOAM_SMARTREDIS/install/lib
export FOAM_SMARTREDIS_BUILD_LIB=$FOAM_SMARTREDIS/build
export LD_LIBRARY_PATH=$FOAM_SMARTREDIS_BUILD_LIB:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$FOAM_SMARTREDIS_LIB:$LD_LIBRARY_PATH
export SSDB="127.0.0.1:8000" # for multinode setup let smartsim do this
