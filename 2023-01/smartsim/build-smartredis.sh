#!/usr/bin/env bash

# Run configure-smartredis.sh first.

# GCC 5-9 is recommended. There are known bugs with GCC >= 10.
# https://www.craylabs.org/docs/installation.html

# OpenFOAM requirement: build OpenFOAM with the same compiler as smartredis. 
# In the file $WM_PROJECT_DIR/wmake/rules/General/Gcc/c++
# use 
# CC          = g++-9 -std=c++2a

export CC=gcc-9
export CXX=g++-9
export NO_CHECKS=1 # skip build checks

smart clobber 
smart build --device=cpu

cd $FOAM_SMARTREDIS && make lib && cd .
