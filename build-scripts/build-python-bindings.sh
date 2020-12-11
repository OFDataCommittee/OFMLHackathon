#!/bin/bash

cd ./src/python

# remove existing build
rm -rf build

# create build dir
mkdir build
cd build

# TODO platform dependendant built step
cmake ..
make

# move python module to module directory
cp silcPy.cpython-*.so ../module/silc/