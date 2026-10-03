cmake_minimum_required(VERSION 3.18)
project(blurfft VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

# The core is header-only. Floating-point contraction (fusing a*b + c into one
# rounding) is turned off so every operation rounds exactly as written: that is
# what makes hardware float and _Float16 results bit-identical to the emulated
# formats, and the precision experiments fair.
add_library(blurfft_core INTERFACE)
target_include_directories(blurfft_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include)
target_link_libraries(blurfft_core INTERFACE Threads::Threads)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(blurfft_core INTERFACE -ffp-contract=off)
  # GCC may otherwise keep _Float16 arithmetic in float between operations.
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag(-fexcess-precision=standard BLURFFT_HAS_EXCESS_PRECISION_FLAG)
  if(BLURFFT_HAS_EXCESS_PRECISION_FLAG)
    target_compile_options(blurfft_core INTERFACE -fexcess-precision=standard)
  endif()
elseif(MSVC)
  target_compile_options(blurfft_core INTERFACE /fp:precise)
endif()

if(SKBUILD)
  # Built by pip (scikit-build-core): the Python extension module.
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_core MODULE cpp/python/bindings.cpp)
  target_link_libraries(_core PRIVATE blurfft_core)
  install(TARGETS _core DESTINATION blurfft)
else()
  # Built directly with CMake: the C++ tests and the command-line benchmark.
  enable_testing()
  add_executable(blurfft_tests cpp/tests/test_core.cpp)
  target_link_libraries(blurfft_tests PRIVATE blurfft_core)
  add_test(NAME blurfft_tests COMMAND blurfft_tests)
endif()
