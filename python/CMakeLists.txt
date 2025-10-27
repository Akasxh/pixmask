find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_pixmask MODULE bindings.cpp)

target_link_libraries(_pixmask PRIVATE pixmask)

target_include_directories(
    _pixmask
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

set_target_properties(
    _pixmask
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/pixmask"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/pixmask"
)

install(
    TARGETS _pixmask
    LIBRARY DESTINATION pixmask
    ARCHIVE DESTINATION pixmask
    RUNTIME DESTINATION pixmask
)
