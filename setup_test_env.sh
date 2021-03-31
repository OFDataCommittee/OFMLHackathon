#!/bin/bash


# setup the necessary environment variables for testing and builds
# this must be *sourced* in the top level smartsim directory in the
# shell that will be used for building.

echo "Setting up SmartRedis environment for testing"

source ./setup_env.sh

export SMARTREDIS_TEST_CLUSTER=True
echo SMARTREDIS_TEST_CLUSTER set to $SMARTREDIS_TEST_CLUSTER

