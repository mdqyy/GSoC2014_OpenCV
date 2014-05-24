cmake_minimum_required(VERSION 2.8.3)
project(matlab)
include(../cmake/utils.cmake)

find_package(MATLAB REQUIRED QUIET)

add_subdirectory(../algorithms ${CMAKE_CURRENT_BINARY_DIR}/libs)
include_directories(${ALG_INCLUDE_DIRS})

add_library(structuredEdgeDetection MODULE structuredEdgeDetection.cpp thirdparty/MxArray.cpp)

target_link_libraries(structuredEdgeDetection ${ALG_LIBS} ${OpenCV_LIBS} ${MATLAB_LIBS})

set_target_properties(structuredEdgeDetection
                      PROPERTIES SUFFIX "${MATLAB_MEXEXT}")

set_target_properties(structuredEdgeDetection
                      PROPERTIES PREFIX "")

if (NOT DEFINED copyTo)
    set(OUTPUT_COPY_TO "" CACHE STRING "binaries are copied to this directory during post-build" FORCE)
endif()
if (NOT "${copyTo}" STREQUAL "")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${copyTo}" CACHE STRING
        "The directory into which runtime target files are built" FORCE)
endif()