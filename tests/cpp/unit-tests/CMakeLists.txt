project(CppClientUnitTester)

cmake_minimum_required(VERSION 3.10)

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

set(SOURCES
	../../../src/cpp/addressanycommand.cpp
	../../../src/cpp/addressatcommand.cpp
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

add_executable(cpp_unit_tests ${SOURCES} ${UNIT_TESTS})

target_link_libraries(cpp_unit_tests ${SR_LIB})
