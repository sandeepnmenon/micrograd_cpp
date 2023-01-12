cmake_minimum_required(VERSION 3.15.0)
project(micrograd_cpp VERSION 0.1.0)

include(CTest)
enable_testing()

add_library(micrograd_cpp INTERFACE)
target_include_directories(micrograd_cpp INTERFACE include)

add_subdirectory(tests)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
