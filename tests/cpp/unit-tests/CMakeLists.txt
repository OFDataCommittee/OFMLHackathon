# BSD 2-Clause License
#
# Copyright (c) 2021-2022, Hewlett Packard Enterprise
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

project(CppClientUnitTester)

cmake_minimum_required(VERSION 3.13)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_BUILD_TYPE DEBUG)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS --coverage)

find_library(SR_LIB smartredis PATHS ../../../install/lib NO_DEFAULT_PATH REQUIRED)

include_directories(SYSTEM
    /usr/local/include
    ../../../include
	../../../install/include
)

# The source files to assess coverage on
set(SOURCES
	../../../src/cpp/addressanycommand.cpp
	../../../src/cpp/addressatcommand.cpp
	../../../src/cpp/addressallcommand.cpp
	../../../src/cpp/client.cpp
	../../../src/cpp/clusterinfocommand.cpp
	../../../src/cpp/command.cpp
	../../../src/cpp/commandlist.cpp
	../../../src/cpp/commandreply.cpp
	../../../src/cpp/compoundcommand.cpp
	../../../src/cpp/dataset.cpp
	../../../src/cpp/dbinfocommand.cpp
	../../../src/cpp/dbnode.cpp
	../../../src/cpp/gettensorcommand.cpp
	../../../src/cpp/keyedcommand.cpp
	../../../src/cpp/metadata.cpp
	../../../src/cpp/metadatafield.cpp
	../../../src/cpp/multikeycommand.cpp
	../../../src/cpp/nonkeyedcommand.cpp
	../../../src/cpp/redis.cpp
	../../../src/cpp/rediscluster.cpp
	../../../src/cpp/redisserver.cpp
	../../../src/cpp/singlekeycommand.cpp
	../../../src/cpp/stringfield.cpp
	../../../src/cpp/tensorbase.cpp
	../../../src/cpp/tensorpack.cpp
)

# The unit tests to be executed
set(UNIT_TESTS
	main.cpp
	test_client.cpp
	test_client_ensemble.cpp
	test_dataset.cpp
	test_tensorbase.cpp
	test_dbnode.cpp
	test_commandlist.cpp
	test_tensor.cpp
	test_metadata.cpp
	test_stringfield.cpp
	test_ssdb.cpp
	test_commandreply.cpp
	test_singlekeycommand.cpp
	test_addressatcommand.cpp
	test_compoundcommand.cpp
	test_multikeycommand.cpp
	test_addressanycommand.cpp
	test_dbinfocommand.cpp
	test_clusterinfocommand.cpp
    test_redisserver.cpp
)

# Compare the number of source files given above to the number of source CPP
# files in the directory. If the two numbers aren't equal, we're not testing
# coverate completely. It is not recommended to use globbed file lists
# in add_executable()
list(LENGTH SOURCES source_count)

file(GLOB CPP_SOURCE_FILES ../../../src/cpp/*.cpp)

list(LENGTH CPP_SOURCE_FILES cpp_source_file_count)

if(NOT (source_count EQUAL cpp_source_file_count))
	message(FATAL_ERROR "CPP test coverage sources count (${source_count}) \
does not match count of cpp source files: ${cpp_source_file_count}")
endif()

# Compare the number of unit test files given above to the number of unit
# test CPP files in the directory. If the two numbers aren't equal, we're
# not running all the tests! It is not recommended to use globbed file lists
# in add_executable()
list(LENGTH UNIT_TESTS unit_test_count)

file(GLOB UNIT_TEST_FILES *.cpp)

list(LENGTH UNIT_TEST_FILES cpp_unit_test_file_count)

if(NOT (unit_test_count EQUAL cpp_unit_test_file_count))
	message(FATAL_ERROR "CPP unit test count (${unit_test_count}) \
does not match count of cpp unit test files: ${cpp_unit_test_file_count}")
endif()

add_executable(cpp_unit_tests ${SOURCES} ${UNIT_TESTS})

target_link_libraries(cpp_unit_tests ${SR_LIB})
