#!/bin/bash


# setup the necessary environment variables for testing and builds
# this must be *sourced* in the top level smartsim directory in the
# shell that will be used for building.

echo "Setting up SILC environment for testing"

source ./setup_env.sh

export SILC_TEST_CLUSTER=False
echo SILC_TEST_CLUSTER set to $SILC_TEST_CLUSTER
