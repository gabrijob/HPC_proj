cmake_minimum_required(VERSION 3.1)


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake/modules/")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/bin")
set(TF_DIR $ENV{TF_PATH}) 
set(ABSL_DIR $ENV{ABSL_INSTALL_PATH}) 

project(HPC_proj)

include_directories(${Protobuf_INCLUDE_DIRS})

find_library(TF_FW_LIB tensorflow_framework HINTS  "${TF_DIR}/bazel-bin/tensorflow")
find_library(TF_LIB tensorflow_cc HINTS "${TF_DIR}/bazel-bin/tensorflow")
find_library(PROTOBUF_LIB protobuf HINTS "/usr/local/lib/")

add_executable(CatDogCNN_simple "")

target_link_libraries(CatDogCNN_simple PUBLIC "${TF_LIB}" "${TF_FW_LIB}" "${PROTOBUF_LIB}")
target_include_directories(CatDogCNN_simple PUBLIC "${TF_DIR}/bazel-tensorflow" "${TF_DIR}/bazel-bin/tensorflow" "${TF_DIR}/bazel-tensorflow/external/eigen_archive" "${TF_DIR}/bazel-tensorflow/third_party" "${ABSL_DIR}")

find_package(OpenMP REQUIRED)
add_executable(CatDogCNN_kfold_cv "")

target_link_libraries(CatDogCNN_kfold_cv PUBLIC "${TF_LIB}" "${TF_FW_LIB}" "${PROTOBUF_LIB}" "${OpenMP_CXX_FLAGS}")
target_include_directories(CatDogCNN_kfold_cv PUBLIC "${TF_DIR}/bazel-tensorflow" "${TF_DIR}/bazel-bin/tensorflow" "${TF_DIR}/bazel-tensorflow/external/eigen_archive" "${TF_DIR}/bazel-tensorflow/third_party" "${ABSL_DIR}")
target_compile_options(CatDogCNN_kfold_cv PRIVATE "${OpenMP_CXX_FLAGS}")
add_subdirectory(${PROJECT_SOURCE_DIR}/src)