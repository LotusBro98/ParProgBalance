cmake_minimum_required(VERSION 3.0)
project(static_balance)
project(dynamic_balance)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_COMPILER mpic++)

include_directories(/usr/include/mpich)

add_executable(static_balance static.cpp)
add_executable(dynamic_balance dynamic.cpp)