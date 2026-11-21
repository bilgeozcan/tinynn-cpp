cmake_minimum_required(VERSION 3.14)
project(tinynn VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(tinynn INTERFACE)
target_include_directories(tinynn INTERFACE include)
if(MSVC)
  target_compile_options(tinynn INTERFACE /W4)
else()
  target_compile_options(tinynn INTERFACE -Wall -Wextra -Wpedantic)
endif()

add_executable(xor examples/xor.cpp)
target_link_libraries(xor PRIVATE tinynn)

add_executable(regression examples/regression.cpp)
target_link_libraries(regression PRIVATE tinynn)

enable_testing()
add_executable(test_tinynn tests/test_tinynn.cpp)
target_link_libraries(test_tinynn PRIVATE tinynn)
add_test(NAME tinynn COMMAND test_tinynn)
