cmake_minimum_required(VERSION 3.0)
project(static_balance)
project(dynamic_balance)
project(rank)
project(time)
project(sum)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_COMPILER mpic++)

include_directories(/usr/include/mpich)

add_executable(static_balance static.cpp)
add_executable(dynamic_balance dynamic.cpp)
add_executable(rank rank.cpp)
add_executable(time time.cpp)
add_executable(sum sum.cpp)