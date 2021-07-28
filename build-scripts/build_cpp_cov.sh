#!/bin/bash

BASEDIR=$(pwd)
CPP_COV_DIR=$BASEDIR/tests/cpp/unit-tests/build/CMakeFiles/cpp_unit_tests.dir
LCOV_EXEC=$BASEDIR/third-party/lcov/install/usr/local/bin/lcov
GEN_HTML_EXEC=$BASEDIR/third-party/lcov/install/usr/local/bin/genhtml



cd ./CMakeFiles/cpp_unit_tests.dir

EXCLUDE_LIST=""
ENV_VAR=$(echo "$SMARTREDIS_TEST_CLUSTER" | tr '[:upper:]' '[:lower:]')
case $ENV_VAR in
    "false")
        EXCLUDE_LIST="*rediscluster.cpp*"
        ;;
    "true")
        EXCLUDE_LIST="*redis.cpp*"
        ;;
esac

$LCOV_EXEC --capture --directory $CPP_COV_DIR --output-file $CPP_COV_DIR/coverage.info --exclude "$EXCLUDE_LIST" --include  "$BASEDIR/src*" --include "$BASEDIR/include*"
$GEN_HTML_EXEC $CPP_COV_DIR/coverage.info --output-directory $CPP_COV_DIR/../../htmlcov

echo "Coverage information available in file: $CPP_COV_DIR/../../htmlcov/index.html"

cd $BASEDIR