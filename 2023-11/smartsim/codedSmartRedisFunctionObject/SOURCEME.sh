#!/usr/bin/bash

current_dir=$PWD
export REPO_ROOT=$(git rev-parse --show-toplevel)
export FOAM_SMARTREDIS="$REPO_ROOT/2023-01/smartsim/smartredis"
cd "$REPO_ROOT" || exit 1
git subtree pull --prefix 2023-01/smartsim/smartredis https://github.com/CrayLabs/SmartRedis develop --squash
cd "$FOAM_SMARTREDIS"
make lib
cd "$current_dir"
if [ ! -d "$FOAM_SMARTREDIS" ]; then
    echo "$FOAM_SMARTREDIS does not exist, please source SOURCEME.sh from its folder"
fi
export SMARTREDIS_INCLUDE=$FOAM_SMARTREDIS/install/include
export SMARTREDIS_LIB=$FOAM_SMARTREDIS/install/lib
export LD_LIBRARY_PATH=$SMARTREDIS_LIB:$LD_LIBRARY_PATH
export SSDB="127.0.0.1:8000" 
export FOAM_CODE_TEMPLATES=$REPO_ROOT/2023-11/smartsim/codedSmartRedisFunctionObject/dynamicCode/

wmake libso clientWrapper
wmake libso src
