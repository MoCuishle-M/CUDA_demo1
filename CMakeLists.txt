cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(CUDADemo1 LANGUAGES CXX CUDA)

find_package(CUDAToolkit 12.5 REQUIRED)
## 查找 Google Test
#find_package(GTest REQUIRED)
# 需下载安装cuda toolkit，直接搜即可
set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})

# set compiler flags
set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler -Wall")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  \
                        -gencode=arch=compute_70,code=\\\"sm_70,compute_70\\\" \
                        -gencode=arch=compute_75,code=\\\"sm_75,compute_75\\\" \
                        -gencode=arch=compute_80,code=\\\"sm_80,compute_80\\\" \
                        -gencode=arch=compute_86,code=\\\"sm_86,compute_86\\\" \
                        -gencode=arch=compute_89,code=\\\"sm_89,compute_89\\\" \
                        ")
# Enable optimization at the project level
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2")

# GPU Compute Capability
# https://developer.nvidia.com/cuda-gpus
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89)
#输出的可执行文件保存地址
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(COMMON_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/15_gemv
        ${CUDA_PATH}/include
)

#set(COMMON_LIB_DIRS
#        ${CUDA_PATH}/lib/x64
#)
#指定头文件路径
include_directories(
        ${COMMON_HEADER_DIRS}
)
## 将 Google Test 添加到项目
#include_directories(${GTEST_INCLUDE_DIRS})

##指定待链接的lib路径
#link_directories(
#        ${COMMON_LIB_DIRS}
#)

# set cuda flags
set(CMAKE_CUDA_STANDARD 17)

# example 1
add_executable(1_hello_cuda 1_hello_cuda.cu)
target_link_libraries(
        1_hello_cuda PUBLIC ${CUDA_LIBRARIES} cublas cudart cudadevrt)

# example 2
add_executable(2_indexing 2_indexing.cu)
target_link_libraries(
        2_indexing PUBLIC
        -lcudart
        -lcudadevrt)

# example 3
add_executable(3_vectorAdd 3_vectorAdd.cu)
target_link_libraries(
        3_vectorAdd PUBLIC
        -lcudart
        -lcudadevrt)

# example 3_1
add_executable(3_vectorAdd2 3_vectorAdd2.cu)
target_link_libraries(
        3_vectorAdd2 PUBLIC
        -lcudart
        -lcudadevrt)

# example 3_2
add_executable(3_vectorAdd_vectorized 3_vectorAdd_vectorized.cu)
target_link_libraries(
        3_vectorAdd_vectorized PUBLIC
        -lcudart
        -lcudadevrt)

# example 4
add_executable(4_device_query 4_device_query.cu)
target_link_libraries(
        4_device_query PUBLIC
        -lcudart
        -lcudadevrt)
#5.reduce
add_executable(reduce_baseline
        5_reduce/reduce_baseline.cu
)
target_link_libraries(
        reduce_baseline PUBLIC
        -lcudart
        -lcudadevrt)
add_executable(reduce_v0
        5_reduce/reduce_v0.cu
)
target_link_libraries(
        reduce_v0 PUBLIC
        -lcudart
        -lcudadevrt)
add_executable(reduce_v1
        5_reduce/reduce_v1.cu
)
target_link_libraries(
        reduce_v1 PUBLIC
        -lcudart
        -lcudadevrt)
add_executable(reduce_v2
        5_reduce/reduce_v2.cu
)
target_link_libraries(
        reduce_v2 PUBLIC
        -lcudart
        -lcudadevrt)
add_executable(reduce_v3
        5_reduce/reduce_v3.cu
)
target_link_libraries(
        reduce_v3 PUBLIC
        -lcudart
        -lcudadevrt)
add_executable(reduce_v4
        5_reduce/reduce_v4.cu
)
target_link_libraries(
        reduce_v4 PUBLIC
        -lcudart
        -lcudadevrt)
add_executable(reduce_v5
        5_reduce/reduce_v5.cu
)
target_link_libraries(
        reduce_v5 PUBLIC
        -lcudart
        -lcudadevrt)
add_executable(reduce_v6
        5_reduce/reduce_v6.cu
)
target_link_libraries(
        reduce_v6 PUBLIC
        -lcudart
        -lcudadevrt)
add_executable(redeuce_v7 5_reduce/reduce_v7.cu)
target_link_libraries(
        redeuce_v7 PUBLIC
        -lcudart
        -lcudadevrt)


# example 6
add_executable(6_warp_level_reduce
        6_warp_level_reduce.cu)
target_link_libraries(
        6_warp_level_reduce PUBLIC
        -lcudart
        -lcudadevrt)

# example 7
add_executable(7_histogram
        7_histogram.cu)
target_link_libraries(
        7_histogram PUBLIC
        -lcudart
        -lcudadevrt)

add_executable(7_1_histogram
        7_1_histogram.cu)
target_link_libraries(
        7_1_histogram PUBLIC
        -lcudart
        -lcudadevrt)

add_executable(7_1_1_histogram
        7_1_1_histogram.cu)
target_link_libraries(
        7_1_1_histogram PUBLIC
        -lcudart
        -lcudadevrt)

add_executable(7_2_histogram
        7_2_histogram.cu)
target_link_libraries(
        7_2_histogram PUBLIC
        -lcudart
        -lcudadevrt)

# example 8
add_executable(8_copy_if
        8_copy_if.cu)
target_link_libraries(
        8_copy_if PUBLIC
        -lcudart
        -lcudadevrt)

# example 9
add_executable(9_gelu
        9_gelu.cu)
target_link_libraries(
        9_gelu PUBLIC
        -lcudart
        -lcudadevrt)

# example 10
add_executable(10_fused_bias_mask_scale_and_add_fp16 10_fused_bias_mask_scale_and_add/10_fused_bias_mask_scale_and_add_fp16.cu)
target_link_libraries(
        10_fused_bias_mask_scale_and_add_fp16 PUBLIC
        -lcudart
        -lcudadevrt
)
add_executable(10_fused_bias_mask_scale_and_add_fp32 10_fused_bias_mask_scale_and_add/10_fused_bias_mask_scale_and_add_fp32.cu)
target_link_libraries(
        10_fused_bias_mask_scale_and_add_fp32 PUBLIC
        -lcudart
        -lcudadevrt
)

# example oneflow_elementwise_test
add_executable(elementwise_test elementwise_test.cu)
target_link_libraries(
        elementwise_test PUBLIC
        -lcudart
        -lcudadevrt
)
# 为elementwise_test启用扩展对齐支持
target_compile_definitions(elementwise_test PRIVATE _ENABLE_EXTENDED_ALIGNED_STORAGE)

#11
add_executable(11_softmax
        11_softmax.cu
)
target_link_libraries(
        11_softmax PUBLIC
        -lcudart
        -lcudadevrt)

#11
add_executable(11_2_oneflow_softmax
        11_2_oneflow_softmax.cu
)
target_link_libraries(
        11_2_oneflow_softmax PUBLIC
        -lcudart
        -lcudadevrt)
#12
add_executable(12_measure_GPU_peak_perf
        12_measure_GPU_peak_perf.cu
)
target_link_libraries(
        12_measure_GPU_peak_perf PUBLIC
        -lcudart
        -lcudadevrt)
#13 CUDA stream
add_executable(13_1_CUDAstream
        13_CUDAstream/13_1_CUDAstream_kernels_overlap.cu
)
target_link_libraries(
        13_1_CUDAstream PUBLIC
        -lcudart
        -lcudadevrt)
add_executable(13_2_CUDAstream
        13_CUDAstream/13_2_CUDAstream_kernel_copy_overlap.cu
)
target_link_libraries(
        13_2_CUDAstream PUBLIC
        -lcudart
        -lcudadevrt)
#14
add_executable(14_quantize
        14_quantize.cu
)
target_link_libraries(
        14_quantize PUBLIC
        -lcudart
        -lcudadevrt)
#15 gemv
add_executable(15_1_gemv
        15_gemv/15_1_fp32_fp16_gemv.cu
)
target_link_libraries(
        15_1_gemv PUBLIC
        -lcudart
        -lcudadevrt)
add_executable(15_2_gemv
        15_gemv/15_2_fp32_fp16_gemv.cu
)
target_link_libraries(
        15_2_gemv PUBLIC
        -lcudart
        -lcudadevrt)

## 添加测试目标
#add_executable(15_test_half_struct 15_gemv/15_test_half_struct.cu)
## 链接 Google Test 库
#target_link_libraries(15_test_half_struct PUBLIC
#        ${GTEST_BOTH_LIBRARIES}
#        -lcudart
#        -lcudadevrt)

#16 dropout
add_executable(16_dropout
        16_fused_dropout/16_dropout.cu
)
target_link_libraries(
        16_dropout PUBLIC
        -lcudart
        -lcudadevrt)