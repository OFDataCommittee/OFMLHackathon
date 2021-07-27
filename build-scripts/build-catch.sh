#!/bin/bash

# Install Catch
if [[ -f ./catch/catch.hpp ]]; then
    echo "Catch has already been download and installed"
else
    echo "Installing Catch"
    if [[ ! -d "./catch" ]]; then
        wget https://github.com/catchorg/Catch2/releases/download/v2.13.6/catch.hpp -P catch
    else
        echo "Catch downloaded"
    fi
    echo "Finished installing Catch"
fi
