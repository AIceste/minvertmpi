cmake_minimum_required (VERSION 2.6)
set(CMAKE_C_COMPILER "mpicc")
set(CMAKE_CXX_COMPILER "mpic++")

project (tp3)

add_executable(tp3 main.cpp Matrix.cpp)

INCLUDE(CPack)
