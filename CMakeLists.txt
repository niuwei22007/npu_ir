cmake_minimum_required(VERSION 3.17)
project(build_ir_model)

include_directories(
        ${PROJECT_SOURCE_DIR}/jni
        ${PROJECT_SOURCE_DIR}/ddk/ai_ddk_lib/include
        ${PROJECT_SOURCE_DIR}/ddk/ai_ddk_lib/include/graph
        ${PROJECT_SOURCE_DIR}/../../opt/android-ndk-r20b/sysroot/usr/include
)

file(GLOB SRC0 "${PROJECT_SOURCE_DIR}/jni/*.cpp")
file(GLOB SRC1 "${PROJECT_SOURCE_DIR}/jni/*/*.cpp")

add_library(
        other_cpp
        SHARED
        ${SRC0}
        ${SRC1}
        jni/test_contains_npu.cpp)