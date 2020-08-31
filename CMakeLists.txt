SET(CMAKE_CXX_STANDARD 17)
cmake_minimum_required(VERSION 3.1)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)
project (silc)

set(PROJECT_INCLUDE_DIR    ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(CPP_ROOT_FOLDER_NAME   silc)
include_directories(${PROJECT_INCLUDE_DIR})


# Dependencies
# ============
find_package(xtl      REQUIRED)
find_package(xtensor  REQUIRED)

# # Build
# # =====


SET(INTERFACE_LIB_NAME silc)

file(GLOB_RECURSE ${PROJECT_NAME}_HEADERS  ${PROJECT_INCLUDE_DIR}/*.hpp)

add_library(${INTERFACE_LIB_NAME} INTERFACE)

message(STATUS "INSTALL_INTERFACE: ${CMAKE_INSTALL_INCLUDEDIR}")
target_include_directories(${INTERFACE_LIB_NAME}  INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)



target_link_libraries(${INTERFACE_LIB_NAME}
  INTERFACE xtensor)


# Python
# ============
OPTION(BUILD_PYTHON "${PROJECT_NAME} python binding" ON)
if(BUILD_PYTHON)
    add_subdirectory(python)
endif()



# Tests
###########

OPTION(BUILD_TESTS "${PROJECT_NAME} test suite" ON)
# OPTION(DOWNLOAD_DOCTEST "build doctest from downloaded sources" ON)

# if(DOWNLOAD_DOCTEST)
#     set(BUILD_TESTS ON)
# endif()

if(BUILD_TESTS)
    add_subdirectory(test)
endif()



# Installation
# ============

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)


set(${PROJECT_NAME}_CMAKECONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}" CACHE
    STRING "install path for ${PROJECT_NAME}Config.cmake")


message(STATUS "DEST ${${PROJECT_NAME}_CMAKECONFIG_INSTALL_DIR}" )


install(TARGETS ${INTERFACE_LIB_NAME}
        EXPORT ${INTERFACE_LIB_NAME}-targets)

install(EXPORT ${INTERFACE_LIB_NAME}-targets
        FILE ${INTERFACE_LIB_NAME}Targets.cmake
        #NAMESPACE ${PROJECT_NAME}::
        DESTINATION lib/cmake/${PROJECT_NAME}
         )


install(DIRECTORY  ${PROJECT_INCLUDE_DIR}/${CPP_ROOT_FOLDER_NAME}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})




configure_package_config_file(${PROJECT_NAME}Config.cmake.in
                              "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
                              INSTALL_DESTINATION ${${PROJECT_NAME}_CMAKECONFIG_INSTALL_DIR})


# ${PROJECT_NAME} is header-only and does not depend on the architecture.
# Remove CMAKE_SIZEOF_VOID_P from ${PROJECT_NAME}ConfigVersion.cmake so that an ${PROJECT_NAME}Config.cmake
# generated for a 64 bit target can be used for 32 bit targets and vice versa.
set(_${PROJECT_NAME}_CMAKE_SIZEOF_VOID_P ${CMAKE_SIZEOF_VOID_P})
unset(CMAKE_SIZEOF_VOID_P)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
                                 VERSION ${${PROJECT_NAME}_VERSION}
                                 COMPATIBILITY AnyNewerVersion)
set(CMAKE_SIZEOF_VOID_P ${_${PROJECT_NAME}_CMAKE_SIZEOF_VOID_P})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        DESTINATION ${${PROJECT_NAME}_CMAKECONFIG_INSTALL_DIR})

install(EXPORT ${PROJECT_NAME}-targets
        FILE ${PROJECT_NAME}Targets.cmake
        DESTINATION ${${PROJECT_NAME}_CMAKECONFIG_INSTALL_DIR})

configure_file(${PROJECT_NAME}.pc.in
               "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc"
                @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc"
DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig/")



