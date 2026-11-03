# Copyright (c) 2022-2026, T-HEAD (SHANGHAI) SEMICONDUCTOR CO., LTD. All rights reserved. 
# Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
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

cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
cmake_policy(SET CMP0112 NEW)

if(cutlass_LOADED)
  # If CUTLASS has been previously fetched and loaded, don't do it again.
  return()
else()
  set(cutlass_LOADED ON)
  set(CUTLASS_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE PATH "CUTLASS Repository Directory")
endif()

message(STATUS "CMake Version: ${CMAKE_VERSION}")

# To reduce duplicate version locations, parse the version out of the
# main versions.h file and reuse it here.

file(READ ${CMAKE_CURRENT_SOURCE_DIR}/include/cutlass/version.h VERSION_FILE_CONTENTS)
string(REGEX MATCH "#define CUTLASS_MAJOR ([0-9]+)" _CUTLASS_VERSION_MAJOR "${VERSION_FILE_CONTENTS}")
set(_CUTLASS_VERSION_MAJOR ${CMAKE_MATCH_1})
string(REGEX MATCH "#define CUTLASS_MINOR ([0-9]+)" _CUTLASS_VERSION_MINOR "${VERSION_FILE_CONTENTS}")
set(_CUTLASS_VERSION_MINOR ${CMAKE_MATCH_1})
string(REGEX MATCH "#define CUTLASS_PATCH ([0-9]+)" _CUTLASS_VERSION_PATCH "${VERSION_FILE_CONTENTS}")
set(_CUTLASS_VERSION_PATCH ${CMAKE_MATCH_1})

message(STATUS "CUTLASS ${_CUTLASS_VERSION_MAJOR}.${_CUTLASS_VERSION_MINOR}.${_CUTLASS_VERSION_PATCH}")

## CUTLASS PROJECT #############################################################

project(CUTLASS VERSION ${_CUTLASS_VERSION_MAJOR}.${_CUTLASS_VERSION_MINOR}.${_CUTLASS_VERSION_PATCH} LANGUAGES CXX)

################################################################################

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/PPUToolchain.cmake)

################################################################################

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX install CACHE PATH "Default installation location." FORCE)
endif()

message(STATUS "Default Install Location: ${CMAKE_INSTALL_PREFIX}")

set(CUTLASS_TEST_LEVEL "0" CACHE STRING "Level of tests to compile.")
# 0 - Sanity, 1 - Release-Quality, 2 - Exhaustive

find_package(Python3 3.5 COMPONENTS Interpreter REQUIRED)

################################################################################
set(CUTLASS_ENABLE_HEADERS_ONLY OFF CACHE BOOL "Enable only the header library")

if(CUTLASS_ENABLE_HEADERS_ONLY)
  set(CUTLASS_ENABLE_EXAMPLES_INIT OFF)
  set(CUTLASS_ENABLE_TOOLS_INIT ON)
  set(CUTLASS_ENABLE_LIBRARY_INIT OFF)
  set(CUTLASS_ENABLE_TESTS_INIT OFF)
else()
  set(CUTLASS_ENABLE_EXAMPLES_INIT ON)
  set(CUTLASS_ENABLE_TOOLS_INIT ON)
  set(CUTLASS_ENABLE_LIBRARY_INIT ON)
  if(${CMAKE_PROJECT_NAME} STREQUAL ${PROJECT_NAME})
    set(CUTLASS_ENABLE_TESTS_INIT ON)
  else()
    set(CUTLASS_ENABLE_TESTS_INIT OFF)
  endif()
endif()

set(CUTLASS_TEST_UNIT_ENABLE_WARNINGS OFF CACHE BOOL "Enable warnings on waived unit tests.")

set(CUTLASS_ENABLE_EXAMPLES ${CUTLASS_ENABLE_EXAMPLES_INIT} CACHE BOOL "Enable CUTLASS Examples")
set(CUTLASS_ENABLE_TOOLS ${CUTLASS_ENABLE_TOOLS_INIT} CACHE BOOL "Enable CUTLASS Tools")
set(CUTLASS_ENABLE_LIBRARY OFF CACHE BOOL "Enable CUTLASS Library")
set(CUTLASS_ENABLE_PROFILER OFF CACHE BOOL "Enable CUTLASS Profiler")

set(CUTLASS_ENABLE_TESTS ${CUTLASS_ENABLE_TESTS_INIT} CACHE BOOL "Enable CUTLASS Tests")
set(CUTLASS_ENABLE_GTEST_UNIT_TESTS ${CUTLASS_ENABLE_TESTS} CACHE BOOL "Enable CUTLASS GTest-based Unit Tests")
set(CUTLASS_USE_SYSTEM_GOOGLETEST OFF CACHE BOOL "Use system/external installation of GTest")
set(CUTLASS_USE_PACKED_TUPLE ON CACHE BOOL "If ON, make cute::tuple be new standard-layout tuple type; if OFF, use the original cute::tuple implementation that is _not_ standard-layout.")
if (CUTLASS_USE_PACKED_TUPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUTLASS_USE_PACKED_TUPLE=1 -DCUTE_USE_PACKED_TUPLE=1")
  list(APPEND CUTLASS_PPU_EXTRA_HGCC_FLAGS "-DCUTLASS_USE_PACKED_TUPLE=1" "-DCUTE_USE_PACKED_TUPLE=1")
  message(STATUS "Make cute::tuple be the new standard-layout tuple type")
else()
  message(STATUS "Use the original cute::tuple implementation that is _not_ standard-layout")
endif()

################################################################################

# CUTLASS_PPU_ARCHS is validated and populated by PPUToolchain.cmake.
set(CUTLASS_PPU_ARCHS_ENABLED ${CUTLASS_PPU_ARCHS} CACHE STRING "The PPU architectures to build code for.")

# Translate PPU-neutral arch IDs into legacy numeric IDs for the Python
# kernel generator (python/cutlass_library) which uses upstream numeric
# compute_capability convention internally.
set(CUTLASS_LIBRARY_GENERATOR_ARCHS "")
foreach(ARCH IN LISTS CUTLASS_PPU_ARCHS_ENABLED)
  if(ARCH STREQUAL "ppu0010")
    list(APPEND CUTLASS_LIBRARY_GENERATOR_ARCHS "80a")
  elseif(ARCH STREQUAL "ppu0015")
    list(APPEND CUTLASS_LIBRARY_GENERATOR_ARCHS "89")
  else()
    list(APPEND CUTLASS_LIBRARY_GENERATOR_ARCHS "${ARCH}")
  endif()
endforeach()

include(GNUInstallDirs)

link_directories(${PPU_SDK_LIB})

###################################################################################################
#
# Configure CMake variables
#
###################################################################################################

message(STATUS "Device Compilation Architectures: ${CUTLASS_PPU_ARCHS_ENABLED}")

if (NOT (CMAKE_BUILD_TYPE OR CONFIGURATION_TYPES))
  # By default we want to build in Release mode to ensure that we're getting best performance.
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose build level" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "RelWithDebInfo" "Release")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if (DEFINED CMAKE_DEBUG_POSTFIX)
  set(CUTLASS_LIBRARY_DEBUG_POSTFIX_INIT ${CMAKE_DEBUG_POSTFIX})
else()
  set(CUTLASS_LIBRARY_DEBUG_POSTFIX_INIT .debug)
endif()
set(CUTLASS_LIBRARY_DEBUG_POSTFIX ${CUTLASS_LIBRARY_DEBUG_POSTFIX_INIT} CACHE STRING "Default postfix value for debug libraries")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUTLASS_VERSIONS_GENERATED")

#
# CUTLASS NAMESPACE
#
set(CUTLASS_NAMESPACE "cutlass" CACHE STRING "Top level namespace of CUTLASS")

set(CUTLASS_ENABLE_F16C OFF CACHE BOOL "Enable F16C x86 extensions in host code.")

################################################################################
#
# CUTLASS generator cmake configuration
#

# Kernel unified filter file

set(KERNEL_FILTER_FILE "" CACHE STRING "KERNEL FILTER FILE FULL PATH")

if (KERNEL_FILTER_FILE AND NOT CUTLASS_LIBRARY_KERNELS)
  # If a kernel filter file is specified, we want to generate and then
  # filter on the entire kernel set, not the default kernel
  # (sub)set. The user may have overridden CUTLASS_LIBRARY_KERNELS, in which
  # case the resulting kernel set will be the intersection of the two
  # options differenced against CUTLASS_LIBRARY_IGNORE_KERNELS.
  set(CUTLASS_LIBRARY_KERNELS_INIT "*")
else()
  set(CUTLASS_LIBRARY_KERNELS_INIT "")
endif()

if (KERNEL_FILTER_FILE)
  get_filename_component(KERNEL_FILTER_FILE "${KERNEL_FILTER_FILE}" ABSOLUTE)
  set(KERNEL_FILTER_FILE "${KERNEL_FILTER_FILE}" CACHE STRING "KERNEL FILTER FILE FULL PATH" FORCE)
endif()

if(KERNEL_FILTER_FILE)
  message(STATUS "Full path of filter file: ${KERNEL_FILTER_FILE}")
endif()

set(CUTLASS_LIBRARY_OPERATIONS "all" CACHE STRING "Comma-delimited list of operation name filters. Default '' means all operations are enabled.")
set(CUTLASS_LIBRARY_KERNELS ${CUTLASS_LIBRARY_KERNELS_INIT} CACHE STRING "Comma-delimited list of kernel name filters. If unspecified, only the largest tile size is enabled. If the string 'all' is specified, all kernels are enabled.")
set(CUTLASS_LIBRARY_IGNORE_KERNELS "" CACHE STRING "Comma-delimited list of kernels to exclude from build. This option ONLY takes effect if CUTLASS_LIBRARY_KERNELS is set.")
set(CUTLASS_LIBRARY_EXCLUDE_KERNELS "" CACHE STRING "Comma-delimited list of kernels to exclude from build. This option always takes effect, whether or not CUTLASS_LIBRARY_KERNELS is set. It also can exclude kernels from the filter file (see KERNEL_FILTER_FILE).")
set(CUTLASS_LIBRARY_INSTANTIATION_LEVEL "" CACHE STRING "Instantiation level for PPU kernels. Set to `max` and make sure CUTLASS_LIBRARY_KERNELS is non-empty to stamp all possible kernel configurations.")

################################################################################

set(CUTLASS_TEST_ENABLE_CACHED_RESULTS ON CACHE BOOL "Enable caching and reuse of test results in unit tests")

set_property(CACHE CUTLASS_TEST_LEVEL PROPERTY STRINGS 0 1 2)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUTLASS_TEST_LEVEL=${CUTLASS_TEST_LEVEL}")
list(APPEND CUTLASS_PPU_EXTRA_HGCC_FLAGS "-DCUTLASS_TEST_LEVEL=${CUTLASS_TEST_LEVEL}")

if (CUTLASS_TEST_ENABLE_CACHED_RESULTS)
  message(STATUS "Enable caching of reference results in conv unit tests")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1")
  list(APPEND CUTLASS_PPU_EXTRA_HGCC_FLAGS "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1")
endif()

set(CUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED ON CACHE BOOL "Enable/Disable rigorous conv problem sizes in conv unit tests")

if (CUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED)
  message(STATUS "Enable rigorous conv problem sizes in conv unit tests")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1")
  list(APPEND CUTLASS_PPU_EXTRA_HGCC_FLAGS "-DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1")
endif()

################################################################################

# Trace levels for debugging
set(CUTLASS_DEBUG_TRACE_LEVEL "0" CACHE STRING "Level of debug tracing to perform.")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUTLASS_DEBUG_TRACE_LEVEL=${CUTLASS_DEBUG_TRACE_LEVEL}")
list(APPEND CUTLASS_PPU_EXTRA_HGCC_FLAGS "-DCUTLASS_DEBUG_TRACE_LEVEL=${CUTLASS_DEBUG_TRACE_LEVEL}")

set(CUTLASS_SKIP_REDUCTION_INIT OFF CACHE BOOL "Disable init reduction workspace")


###################################################################################################
#
# Configure device-side build options
#
###################################################################################################

if (CUTLASS_SKIP_REDUCTION_INIT)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUTLASS_SKIP_REDUCTION_INIT=1")
  list(APPEND CUTLASS_PPU_EXTRA_HGCC_FLAGS "-DCUTLASS_SKIP_REDUCTION_INIT=1")
endif()

set(CUTLASS_PROFILER_DISABLE_REFERENCE OFF CACHE BOOL "Disable compilation of reference kernels in the CUTLASS profiler.")
if (CUTLASS_PROFILER_DISABLE_REFERENCE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUTLASS_PROFILER_DISABLE_REFERENCE=1")
  list(APPEND CUTLASS_PPU_EXTRA_HGCC_FLAGS "-DCUTLASS_PROFILER_DISABLE_REFERENCE=1")
endif()

set(CUTLASS_ENABLE_SYNCLOG OFF CACHE BOOL "Enable synchronization event logging for race condition debugging. WARNING: This redefines __syncthreads() and __syncwarp() in all downstream code!")

if (CUTLASS_ENABLE_SYNCLOG)
  string(APPEND CMAKE_CXX_FLAGS " -DCUTLASS_ENABLE_SYNCLOG=1")
  list(APPEND CUTLASS_PPU_EXTRA_HGCC_FLAGS "-DCUTLASS_ENABLE_SYNCLOG=1")
endif()

# Warning suppressions for host compilation (g++).
if (CUTLASS_PPU_HOST_COMPILE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-conversion -Wno-unused-parameter -Wno-strict-aliasing")
endif()

if (CUTLASS_ENABLE_F16C AND NOT CMAKE_CROSSCOMPILING)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUTLASS_ENABLE_F16C=1 -mf16c")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wconversion -fno-strict-aliasing")

if (CUTLASS_PPU_DEVICE_COMPILE)
  if (NOT CUTLASS_PPU_HOST_COMPILE)
    message(FATAL_ERROR "PPU device compilation requires the PPU toolchain (PPUToolchain.cmake).")
  endif()

  # Force RPATH (not RUNPATH) so libraries co-located with the PPU
  # runtime can be found at load time.
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--disable-new-dtags")

  link_libraries(ppu::runtime)
  link_libraries(ppu::driver)
endif()

message(STATUS "Host CXX flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Device hgcc flags: ${CUTLASS_PPU_EXTRA_HGCC_FLAGS}")

function(cutlass_apply_standard_compile_options TARGET)
  # Device flags are applied via add_custom_command in cutlass_build_dev_kernels
  # (PPUToolchain.cmake).  This function only links the CUTLASS header target.
  target_link_libraries(${TARGET} PRIVATE CUTLASS)
endfunction()

#
# The following items should eventually be pushed into cutlass/CMakeLists.txt
#

# GLOB for CUTLASS header files. Should we use a static list instead?
file(GLOB_RECURSE CUTLASS_INCLUDE RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} include/cutlass/*.h)
file(GLOB_RECURSE CUTLASS_CUTLASS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/include include/cutlass/*.h include/cutlass/*.hpp include/cutlass/*.inl)
file(GLOB_RECURSE CUTLASS_CUTE RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/include include/cute/*.h*)

# the upstream test/unit/kernel/* tree does not exist in this
# tree, so the GLOB_RECURSE produced an empty list and the variable
# was unused. Kept as a comment so future merges can notice the removal.

###################################################################################################
#
# Define build targets
#
###################################################################################################

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/include REGULAR_EXPRESSION ".*\.h")

add_library(CUTLASS INTERFACE)
add_library(cutlass3::cutlass ALIAS CUTLASS)
set_target_properties(CUTLASS PROPERTIES EXPORT_NAME cutlass)

set(CUTLASS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include CACHE PATH "CUTLASS Header Library")

set(CUTLASS_GENERATOR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tools/library CACHE INTERNAL "Location of generator scripts")

# The following utility directory is needed even if the tools build is disabled, so it exists here.
set(CUTLASS_TOOLS_UTIL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tools/util/include CACHE INTERNAL "")

include_directories(${CUTLASS_INCLUDE_DIR})

# Populate device-compile include flags for hgcc add_custom_command.
# These are passed explicitly because add_custom_command does not inherit
# target include directories.  All commonly-needed project-wide paths
# are included here.
set(CUTLASS_PPU_DEV_INCLUDE_FLAGS
    "-I${CMAKE_CURRENT_SOURCE_DIR}/include"
    "-I${CMAKE_CURRENT_BINARY_DIR}/include"
    "-I${CUTLASS_TOOLS_UTIL_INCLUDE_DIR}"
    "-I${CMAKE_CURRENT_SOURCE_DIR}/tools/library/include"
    "-I${CMAKE_CURRENT_SOURCE_DIR}/tools/library/src"
    "-I${CMAKE_CURRENT_SOURCE_DIR}/examples/common"
    "-I${CMAKE_CURRENT_SOURCE_DIR}/test/unit/common"
    "-I${CMAKE_CURRENT_SOURCE_DIR}/tools/profiler/include"
    "-I${PPU_SDK_INCLUDE}"
    "-I${PPU_SDK_TARGETS_INCLUDE}"
    CACHE INTERNAL "Include flags (-I...) for hgcc device compilation")

target_compile_features(CUTLASS INTERFACE cxx_std_11)

if (NOT CUTLASS_NAMESPACE STREQUAL "cutlass")
  target_compile_definitions(CUTLASS INTERFACE CUTLASS_NAMESPACE=${CUTLASS_NAMESPACE})
endif()

if (NOT DEFINED CUTLASS_REVISION)

  find_package(Git QUIET)

  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
    RESULT_VARIABLE CUTLASS_REVISION_RESULT
    OUTPUT_VARIABLE CUTLASS_REVISION
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  if (CUTLASS_REVISION_RESULT)
    message(STATUS "CUTLASS Revision: Unable to detect, Git returned code ${CUTLASS_REVISION_RESULT}.")
  else()
    message(STATUS "CUTLASS Revision: ${CUTLASS_REVISION}")
  endif()

endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/version_extended.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/cutlass/version_extended.h
  @ONLY)

target_include_directories(
  CUTLASS
  INTERFACE
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CUTLASS_INCLUDE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  )

# Mark SDK headers as system to suppress warnings from them
target_include_directories(
  CUTLASS
  SYSTEM INTERFACE
  $<BUILD_INTERFACE:${PPU_SDK_ROOT}/include>
  )

install(
  DIRECTORY
  ${CUTLASS_INCLUDE_DIR}/
  ${CMAKE_CURRENT_BINARY_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

install(
  TARGETS CUTLASS
  EXPORT Cutlass3
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

################################################################################

# Add common library search paths so executables and libraries can load and run
# without LD_LIBRARY_PATH being set.
link_libraries(
  "-Wl,-rpath,'$ORIGIN'"
  "-Wl,-rpath,'$ORIGIN/../lib64'"
  "-Wl,-rpath,'$ORIGIN/../lib'"
  "-Wl,-rpath,'${PPU_SDK_LIB}'"
  ${CMAKE_DL_LIBS}
  )

################################################################################

include(CTest)
enable_testing()

if (CUTLASS_ENABLE_GTEST_UNIT_TESTS)
  if (CUTLASS_USE_SYSTEM_GOOGLETEST)
    find_package(GTest REQUIRED)
  else()
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/googletest.cmake)
  endif()
  # Append the GTest headers to the hgcc device include flags so that .cu
  # test files compiled via add_custom_command can find <gtest/gtest.h>.
  # For FetchContent builds, googletest_SOURCE_DIR is set by FetchContent.
  # For system installs, use the INTERFACE_INCLUDE_DIRECTORIES of GTest::gtest.
  if (CUTLASS_USE_SYSTEM_GOOGLETEST)
    get_target_property(_gtest_incdirs GTest::gtest INTERFACE_INCLUDE_DIRECTORIES)
    if (_gtest_incdirs)
      foreach(_d IN LISTS _gtest_incdirs)
        list(APPEND CUTLASS_PPU_DEV_INCLUDE_FLAGS "-I${_d}")
      endforeach()
    endif()
  else()
    list(APPEND CUTLASS_PPU_DEV_INCLUDE_FLAGS
         "-I${googletest_SOURCE_DIR}/googletest/include")
  endif()
  set(CUTLASS_PPU_DEV_INCLUDE_FLAGS ${CUTLASS_PPU_DEV_INCLUDE_FLAGS}
      CACHE INTERNAL "Include flags (-I...) for hgcc device compilation" FORCE)
endif()

if (NOT TARGET test_all)
  add_custom_target(test_all)
endif()

set(CUTLASS_INSTALL_TESTS ON CACHE BOOL "Install test executables")
set(CUTLASS_TEST_EXECUTION_ENVIRONMENT "" CACHE BOOL "Environment in which to invoke unit test executables")

set(CMAKE_TEST_INSTALL_PREFIX test CACHE STRING "Test root install location, relative to CMAKE_INSTALL_PREFIX.")
set(CUTLASS_TEST_INSTALL_PREFIX ${CMAKE_TEST_INSTALL_PREFIX}/cutlass CACHE STRING "Test root install location, relative to CMAKE_INSTALL_PREFIX.")
set(CUTLASS_TEST_INSTALL_BINDIR ${CUTLASS_TEST_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR} CACHE STRING "Test root install location, relative to CMAKE_INSTALL_PREFIX.")
set(CUTLASS_TEST_INSTALL_LIBDIR ${CUTLASS_TEST_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR} CACHE STRING "Test root install location, relative to CMAKE_INSTALL_PREFIX.")

install(DIRECTORY DESTINATION ${CUTLASS_TEST_INSTALL_PREFIX})
install(DIRECTORY DESTINATION ${CUTLASS_TEST_INSTALL_BINDIR})
install(DIRECTORY DESTINATION ${CUTLASS_TEST_INSTALL_LIBDIR})
install(DIRECTORY DESTINATION ${CUTLASS_TEST_INSTALL_PREFIX}/ctest)

################################################################################

set(CUTLASS_DEFAULT_ACTIVE_TEST_SETS "default" CACHE STRING "Default
  activated test sets. In `make test` mode, this string determines the
  active set of tests. In `ctest` mode, this value can be overriden
  with CUTLASS_TEST_SETS environment variable when running the ctest
  executable.")

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
set(CUTLASS_CTEST_TEMPLATE_FILE ${CMAKE_CURRENT_LIST_DIR}/cmake/CTestTestfile.configure.cmake)
set(CUTLASS_CTEST_GENERATED_FILES "" CACHE INTERNAL "")

function(cutlass_add_executable_tests NAME TARGET)
#
# Generates test rules for `make test`, `make test_all`, and `ctest` invoked from either the
# <CMAKE_BINARY_DIR> or the <CMAKE_INSTALL_PREFIX>/<CUTLASS_TEST_INSTALL_PREFIX> after installation.
#
# NAME: The base name for the test. Can be run with `make <NAME>` or `ctest -R 'c<NAME>'`.
# TARGET: The target corresponding to the executable under test.
# DISABLE_EXECUTABLE_INSTALL_RULE: An option, if given, that disables creating an install rule for TARGET.
# DEPENDS: A list of targets or files on which this test is dependent.
# DEPENDEES: A list of targets which should depend on this test.
# TEST_COMMAND_OPTIONS: A list of variables (i.e. by reference params) which contain command line arguments
#   to pass to the test executable. A unique test is generated for each set of
#   options given. If this option is not used, a single test with no arguments is generated.
# TEST_COMMAND_OPTIONS_PREFIX: If provided, is added as a prefix to each TEST_COMMAND_OPTIONS value for
#   generating the full variable name to be referenced.
# RESULT_CACHE_FILE: A file to be installed alongside the test executable with pre-computed
#   test results to speed up test runtime.
# TEST_SETS_SUPPORTED: A list of test set names these tests support.
#

  set(options DISABLE_EXECUTABLE_INSTALL_RULE DO_NOT_LOWERCASE_TEST_NAME)
  set(oneValueArgs DISABLE_TESTS RESULT_CACHE_FILE TEST_COMMAND_OPTIONS_PREFIX)
  set(multiValueArgs DEPENDS DEPENDEES TEST_COMMAND_OPTIONS TEST_SETS_SUPPORTED)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT DEFINED __DISABLE_TESTS)
    set(__DISABLE_TESTS OFF)
  endif()

  set(TEST_EXE $<TARGET_FILE_NAME:${TARGET}>)
  set(TEST_EXE_WORKING_DIRECTORY ./${CMAKE_INSTALL_BINDIR})

  if (NOT DEFINED __TEST_SETS_SUPPORTED)
    set(__TEST_SETS_SUPPORTED ${CUTLASS_DEFAULT_ACTIVE_TEST_SETS})
  endif()

  set(TEST_SETS_SUPPORTED ${__TEST_SETS_SUPPORTED})

  if (__RESULT_CACHE_FILE)

    add_custom_command(
      TARGET ${TARGET}
      POST_BUILD
      COMMAND ${CMAKE_COMMAND}
      ARGS -E copy ${__RESULT_CACHE_FILE} "$<TARGET_FILE_DIR:${TARGET}>"
      )

  endif()

  if (NOT __DISABLE_EXECUTABLE_INSTALL_RULE AND CUTLASS_INSTALL_TESTS)

    # file(RELATIVE_PATH CMAKE_CURRENT_BINARY_RELATIVE_DIR ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})

    install(
      TARGETS ${TARGET}
      RUNTIME DESTINATION ${CUTLASS_TEST_INSTALL_BINDIR}
      )

    if (__RESULT_CACHE_FILE)

     install(
       FILES ${__RESULT_CACHE_FILE}
       DESTINATION ${CUTLASS_TEST_INSTALL_BINDIR}/
       )

    endif()

  endif()

  if (NOT __TEST_COMMAND_OPTIONS)
    set(__TEST_COMMAND_OPTIONS " ")
  endif()

  list(LENGTH __TEST_COMMAND_OPTIONS CMD_COUNT)

  if (CMD_COUNT GREATER 1)
    add_custom_target(${NAME} DEPENDS ${TARGET} ${__DEPENDS})
    foreach(DEPENDEE ${__DEPENDEES})
      add_dependencies(${DEPENDEE} ${NAME})
    endforeach()
  endif()

  if (CUTLASS_INSTALL_TESTS)

    set(_INLINE_PER_TEST_CODE)

    file(READ "${PROJECT_SOURCE_DIR}/cmake/CTestTestfile.test.configure.cmake" _INLINE_PER_TEST_CODE_TEMPLATE)

  endif()

  set(TEST_GROUP_NAME ${NAME})

  # To run the tests from an install package with tests enabled, we need to generate test files
  # that don't rely on the current directory structure in build.

  set(TEST_NAME c${NAME})
  set(TEST_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/ctest/${TEST_NAME})
  file(MAKE_DIRECTORY ${TEST_GEN_DIR})

  set(TEST_EXE_PATH $<TARGET_FILE:${TARGET}>)
  set(TEST_USE_EXTENDED_FORMAT ON)
  configure_file("${CUTLASS_CTEST_TEMPLATE_FILE}" "${TEST_GEN_DIR}/CTestTestfile.${TEST_NAME}.cmake" @ONLY)

  set(TEST_EXE_PATH $<TARGET_FILE_NAME:${TARGET}>)
  set(TEST_USE_EXTENDED_FORMAT OFF) # ctest does not support extended add_test format.
  configure_file("${CUTLASS_CTEST_TEMPLATE_FILE}" "${TEST_GEN_DIR}/CTestTestfile.${TEST_NAME}.install.cmake.in" @ONLY)

  foreach(CMD_OPTIONS_VAR IN LISTS __TEST_COMMAND_OPTIONS)

    if (CMD_COUNT GREATER 1)
      set(TESTCASE_NAME "${NAME}_${CMD_OPTIONS_VAR}")
    else()
      set(TESTCASE_NAME "${NAME}")
    endif()

    if (NOT __DO_NOT_LOWERCASE_TEST_NAME)
      string(TOLOWER "${TESTCASE_NAME}" TESTCASE_NAME)
    endif()

    # The following rigmarole is needed to deal with spaces and possible quotes in
    # command line arguments. The options are passed "by reference" as the actual
    # variable names holding the real options. We then expand these in a way that
    # preserves any quotes. Note, they have to be in this order for it to work for
    # all the use cases below.

    set(TEST_COMMAND_OPTIONS ${${__TEST_COMMAND_OPTIONS_PREFIX}${CMD_OPTIONS_VAR}})
    list(JOIN TEST_COMMAND_OPTIONS " " TEST_COMMAND_OPTIONS)
    separate_arguments(TEST_COMMAND_OPTIONS)

    add_custom_target(
      ${TESTCASE_NAME}
      COMMAND
      ${CUTLASS_TEST_EXECUTION_ENVIRONMENT} $<TARGET_FILE:${TARGET}> ${TEST_COMMAND_OPTIONS}
      DEPENDS
      ${TARGET}
      )

    if (CMD_COUNT GREATER 1)
      add_dependencies(${NAME} ${TESTCASE_NAME})
    endif()

    foreach(DEPENDEE ${__DEPENDEES})
      add_dependencies(${DEPENDEE} ${TESTCASE_NAME})
    endforeach()

    set(TESTCASE_NAME c${TESTCASE_NAME})
    string(CONFIGURE "${_INLINE_PER_TEST_CODE_TEMPLATE}" _TEST_CODE @ONLY)
    file(APPEND "${TEST_GEN_DIR}/CTestTestfile.${TEST_NAME}.cmake" "${_TEST_CODE}")
    file(APPEND "${TEST_GEN_DIR}/CTestTestfile.${TEST_NAME}.install.cmake.in" "${_TEST_CODE}")

  endforeach()

  # The following line imports the tests for immediate run via `make test`.

  include(${TEST_GEN_DIR}/CTestTestfile.${TEST_NAME}.cmake)

  set(CUTLASS_CTEST_GENERATED_FILES ${CUTLASS_CTEST_GENERATED_FILES};ctest/${TEST_NAME}/CTestTestfile.${TEST_NAME}.cmake CACHE INTERNAL "")

    if (CUTLASS_INSTALL_TESTS)

    file(GENERATE
      OUTPUT "${TEST_GEN_DIR}/CTestTestfile.${TEST_NAME}.install.cmake"
      INPUT "${TEST_GEN_DIR}/CTestTestfile.${TEST_NAME}.install.cmake.in"
      )

    install(
      FILES "${TEST_GEN_DIR}/CTestTestfile.${TEST_NAME}.install.cmake"
      DESTINATION ${CUTLASS_TEST_INSTALL_PREFIX}/ctest/${TEST_NAME}
      RENAME CTestTestfile.${TEST_NAME}.cmake
      )

    endif()

endfunction()

if (CUTLASS_ENABLE_TOOLS)
  add_subdirectory(tools)
  if (CUTLASS_ENABLE_PROFILER)
    add_dependencies(test_all test_profiler)
  endif()
endif()

if (CUTLASS_ENABLE_EXAMPLES)
  add_subdirectory(examples)
  add_dependencies(test_all test_examples)
endif()

if (CUTLASS_ENABLE_TESTS)
  add_subdirectory(test)
  if (CUTLASS_ENABLE_GTEST_UNIT_TESTS)
  add_dependencies(test_all test_unit)
  endif()
endif()

if (CUTLASS_INSTALL_TESTS)

  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/ctest")

  file(WRITE "${CMAKE_BINARY_DIR}/ctest/CTestTestfile.cmake" "# Generated File\n\n")
  file(APPEND "${CMAKE_BINARY_DIR}/ctest/CTestTestfile.cmake" "cmake_policy(SET CMP0057 NEW) # Allow IN_LIST for if()\n\n")
  file(APPEND "${CMAKE_BINARY_DIR}/ctest/CTestTestfile.cmake" "if (NOT DEFINED ENV{CUTLASS_TEST_SETS})\n")
  file(APPEND "${CMAKE_BINARY_DIR}/ctest/CTestTestfile.cmake" "  set(ENV{CUTLASS_TEST_SETS} ${CUTLASS_DEFAULT_ACTIVE_TEST_SETS})\n")
  file(APPEND "${CMAKE_BINARY_DIR}/ctest/CTestTestfile.cmake" "endif()\n\n")

  foreach(GENERATED_FILE ${CUTLASS_CTEST_GENERATED_FILES})
    file(APPEND "${CMAKE_BINARY_DIR}/ctest/CTestTestfile.cmake" "include(${GENERATED_FILE})\n")
  endforeach()

  install(
    FILES "${CMAKE_BINARY_DIR}/ctest/CTestTestfile.cmake"
    DESTINATION "${CUTLASS_TEST_INSTALL_PREFIX}/"
    )

endif()

################################################################################

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/Cutlass3ConfigVersion.cmake
  COMPATIBILITY AnyNewerVersion)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Cutlass3Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/Cutlass3Config.cmake
  @ONLY
  )

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/Cutlass3Config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/Cutlass3ConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cutlass3/
  )

install(
  EXPORT Cutlass3
  NAMESPACE cutlass3::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Cutlass3/
  FILE Cutlass3Targets.cmake
  )

################################################################################

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Cutlass3PackageConfig.cmake)

