#!/bin/bash

# Install Catch
if [[ -f ./catch/single_include/catch2/catch.hpp ]]; then
    echo "Catch has already been download and installed"
else
    echo "Installing Catch"
    if [[ ! -d "./catch" ]]; then
        git clone https://github.com/catchorg/Catch2.git catch --branch v2.13.6 --depth=1
    else
        echo "Catch downloaded"
    fi
    echo "Finished installing Catch"
fi
