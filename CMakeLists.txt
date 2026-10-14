# mrt2.cpp - CMake build, for Windows (Visual Studio), Linux and macOS.
#
#   cmake -S . -B build
#   cmake --build build --config Release
#   ctest --test-dir build -C Release        (tests; the model tests need the weights)
#
# No -march=native: the fast x86 kernels (AVX2) are chosen at run time
# (cpu.h), so one binary runs on every PC.
cmake_minimum_required(VERSION 3.16)
project(mrt2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Optimized by default: an unoptimized build is ~10x slower than real time.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
  # /utf-8: sources are UTF-8. /permissive-: strict standard C++.
  add_compile_options(/W3 /utf-8 /permissive-)
  # Standard functions like fopen are fine; windows.h must not define min/max.
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
  add_compile_options(-Wall -Wextra)
endif()

find_package(Threads REQUIRED)

set(CORE_SRC weights.cpp safetensors.cpp json.cpp mmap.cpp profile.cpp threads.cpp cpu.cpp)
set(MODEL_SRC ops.cpp quant.cpp blocks.cpp encoder.cpp temporal.cpp depth.cpp conv.cpp codec.cpp)

# Everything but main(): shared by the program, the tools and the tests.
add_library(mrt2core STATIC ${CORE_SRC} ${MODEL_SRC})
target_include_directories(mrt2core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mrt2core PUBLIC Threads::Threads)

# The program. Audio output uses the OS's own API (audio.cpp).
add_executable(mrt2 mrt2.cpp audio.cpp)
target_link_libraries(mrt2 PRIVATE mrt2core)
if(APPLE)
  target_link_libraries(mrt2 PRIVATE "-framework AudioToolbox" "-framework CoreFoundation")
elseif(WIN32)
  target_link_libraries(mrt2 PRIVATE ole32)
else()
  find_package(ALSA REQUIRED) # Debian/Ubuntu: sudo apt install libasound2-dev
  target_link_libraries(mrt2 PRIVATE ALSA::ALSA)
endif()

add_executable(mrt2info mrt2info.cpp)
target_link_libraries(mrt2info PRIVATE mrt2core)
add_executable(mrt2dump mrt2dump.cpp)
target_link_libraries(mrt2dump PRIVATE mrt2core)

# Benchmark: the profiler is compiled into every operation, so it builds
# its own copy of the sources with MRT2_PROFILE.
add_executable(bench tools/bench.cpp ${CORE_SRC} ${MODEL_SRC})
target_include_directories(bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(bench PRIVATE MRT2_PROFILE)
target_link_libraries(bench PRIVATE Threads::Threads)

# ---- tests ------------------------------------------------------------------------
enable_testing()
if(WIN32)
  set(MRT2_HOME "$ENV{USERPROFILE}")
else()
  set(MRT2_HOME "$ENV{HOME}")
endif()
set(MRT2_MODEL "${MRT2_HOME}/models/magenta-rt-v2/checkpoints/mrt2_small.safetensors"
    CACHE FILEPATH "the model, for the tests")
set(MRT2_REF "${MRT2_HOME}/models/magenta-rt-v2/ref/ref_small.safetensors"
    CACHE FILEPATH "Google's reference outputs, for the parity tests (make ref)")

function(mrt2_test name)
  add_executable(${name} tests/${name}.cpp)
  target_link_libraries(${name} PRIVATE mrt2core)
  add_test(NAME ${name} COMMAND ${name} ${ARGN})
endfunction()

# Need nothing.
mrt2_test(test_conv)
mrt2_test(test_ring)
# Need the model.
if(EXISTS "${MRT2_MODEL}")
  mrt2_test(test_mmap "${MRT2_MODEL}")
  mrt2_test(test_weights "${MRT2_MODEL}")
  # Need Google's reference outputs too.
  if(EXISTS "${MRT2_REF}")
    foreach(t test_ops test_encoder test_temporal test_depth test_codec test_threads test_quant)
      mrt2_test(${t} "${MRT2_MODEL}" "${MRT2_REF}")
    endforeach()
  else()
    message(STATUS "mrt2: ${MRT2_REF} not found: parity tests skipped")
  endif()
else()
  message(STATUS "mrt2: ${MRT2_MODEL} not found: model tests skipped")
endif()
