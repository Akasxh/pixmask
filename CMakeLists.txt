cmake_minimum_required(VERSION 3.18)
project(pixmask VERSION 0.1.0 LANGUAGES CXX)

option(PIXMASK_BUILD_PYTHON "Build the pixmask Python bindings" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    add_compile_options(/O2)
else()
    add_compile_options(-O3 -ffast-math)
    if(NOT APPLE)
        add_compile_options(-march=native)
    endif()
endif()

file(GLOB_RECURSE PIXMASK_SOURCES CONFIGURE_DEPENDS
    src/*.cpp
)

add_library(pixmask ${PIXMASK_SOURCES})

target_include_directories(pixmask
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(pixmask PUBLIC cxx_std_17)

add_executable(pixmask_cli tools/pixmask_cli.cpp)
target_link_libraries(pixmask_cli PRIVATE pixmask)

add_executable(pixmask_bench benchmarks/bench.cpp)
target_link_libraries(pixmask_bench PRIVATE pixmask)
target_include_directories(pixmask_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
set_target_properties(pixmask_bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/benchmarks
    OUTPUT_NAME bench)

if(PIXMASK_BUILD_PYTHON)
    add_subdirectory(python)
endif()
