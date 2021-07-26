#!/bin/bash

# Install LCOV
if [[ -f ./lcov/bin/updateversion.pl ]]; then
    echo "LCOV has already been download and installed"
else
    echo "Installing LCOV"
    if [[ ! -d "./lcov" ]]; then
        git clone https://github.com/linux-test-project/lcov.git lcov
        cd lcov
        git checkout tags/v1.15
        cd ..
    else
        echo "LCOV downloaded"
    fi
    cd lcov
    echo "Building LCOV v1.15"
    CC=gcc CXX=g++ make install
    cd ../
fi