#!/usr/bin/env bash

# Generate environment variables for libtorch/

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export OF_TORCH=$SCRIPT_DIR/libtorch/include
export OF_TORCH_INCLUDE=$OF_TORCH/torch/csrc/api/include
export OF_TORCH_LIB=$SCRIPT_DIR/libtorch/lib
