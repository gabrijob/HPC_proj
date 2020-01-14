cmake_minimum_required(VERSION 3.1)

project(V1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(${Protobuf_INCLUDE_DIRS})

find_library(TF_FW_LIB tensorflow_framework HINTS  "lib/")
find_library(TF_LIB tensorflow_cc HINTS "lib/")
find_library(PROTOBUF_LIB protobuf HINTS "lib/")

add_executable(V1 "")

target_link_libraries(V1 PUBLIC "${TF_LIB}" "${TF_FW_LIB}" "${PROTOBUF_LIB}")
target_include_directories(V1 PUBLIC "include" "lib")

add_executable(V2 "")

target_link_libraries(V2 PUBLIC "${TF_LIB}" "${TF_FW_LIB}" "${PROTOBUF_LIB}")
target_include_directories(V2 PUBLIC "include" "lib")

add_subdirectory(${PROJECT_SOURCE_DIR}/src)
