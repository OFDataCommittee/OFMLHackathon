#!/usr/bin/env bash

# Define the environmental variables for SmartRedis 

export FOAM_SMARTREDIS=$HOME/smartredis
export FOAM_SMARTREDIS_INCLUDE=$FOAM_SMARTREDIS/install/include
export FOAM_SMARTREDIS_LIB=$FOAM_SMARTREDIS/install/lib
export LD_LIBRARY_PATH=$FOAM_SMARTREDIS_LIB:$LD_LIBRARY_PATH
