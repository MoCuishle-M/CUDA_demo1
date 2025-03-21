#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <string>
#include <stdexcept>
#include <math.h>

static const char *cudaGetErrorEnum(cudaError_t error) {
    return cudaGetErrorString(error);
}

#define CHECK(call)                                   \
do{                                                   \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

template<typename T>
struct Vec {
    static constexpr int size = 4;
};

template<>
struct Vec<half> {
    static constexpr int size = 8;
};

template<typename T>
struct SumOp {
    __device__ __forceinline__ T operator()(const T &a, const T &b) const { return a + b; }
};

template<>
struct SumOp<half> {
    __device__ __forceinline__ half operator()(const half &a, const half &b) const { return __hadd(a, b); }
};

template<template<typename> class ReductionOp, typename T>
__device__ __forceinline__ T warpReduce(T val) {
    for (int mask = 16; mask > 0; mask >>= 1) {
        val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

// 把block reduce拆分为多个warp reduce来计算
/**
 * 在块级别上执行归约操作。
 * 使用模板参数指定归约操作类型（如加法、最大值等），并使用T类型指定归约值的类型。
 *
 * @tparam ReductionOp 归约操作的模板类。
 * @tparam T 归约值的类型。
 * @param val 需要进行归约的初始值。
 * @return 经过归约操作后的值。
 */
template<template<typename> class ReductionOp, typename T>
__device__ __forceinline__ T blockReduce(T val) {
    // 获取线程在块中的ID。
    int tid = threadIdx.x;
    // 计算线程所属的warps ID。
    int warp_id = tid / 32;
    // 计算线程在warp中的ID
    int lane_id = tid % 32;
    // 向上进1，以防分配的线程数量小于32导致warp nums为0
    int warp_nums = (blockDim.x + 32 - 1) / 32;
    static __shared__ float warpres[64]; // 截至CC9.0 Maximum number of resident threads per SM = 2048; 所以最多有2048/32 =64个warp
    // block内每个warp reduce的结果，该结果保存在每个warp内的0号线程
    val = warpReduce<ReductionOp, T>(val);
    // 如果当前线程是warp中的0号线程，则将warp的归约结果存储到共享内存中
    if (lane_id == 0) {
        warpres[warp_id] = val;
    }
    __syncthreads();
    // 这时候warp归约的结果已经纯在shared memory中，在0 1 2 3......位置
    // 如果当前线程属于活跃的warp，则从共享内存中读取其归约结果；否则，读取0作为初始值。
    // 最后把每个warp的结果再作一个reduce得到最终一个block的结果
    float warp_val = tid < warp_nums ? warpres[tid] : 0;
    return warpReduce<ReductionOp, T>(warp_val);
}

// 一个blk计算一个元素
// mat * vec = {M, N} * {N, 1}/{1, N}
/**
 * @brief 实现矩阵向量乘法。
 *
 * @tparam VECS_PER_THREAD 每个线程处理的pack data数量。
 * @tparam VEC_SIZE 每个pack data的大小。
 *
 */
template<int VECS_PER_THREAD, int VEC_SIZE>
/**
 *
 * @brief 实现矩阵向量乘法。
 *
 * @param matrix 矩阵指针。
 * @param vector 向量指针。
 * @param res 结果指针。
 * @param cols 矩阵的列数。
 *
 */
__global__ void gemv(float *matrix, float *vector, float *res, int cols) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    float thread_local_sum = 0.0f;
    for (int i = 0; i < VECS_PER_THREAD; i++) {
        // 向量化读取matrix和vector，因为是先转成float4指针再读取，所以注意matrix读取的时候，列数需要除以VEC_SIZE
        float4 mat4 = reinterpret_cast<float4 *>(matrix)[bid * (cols / VEC_SIZE) + i * blockDim.x + tid]; // 1 * float4
        float4 vec4 = reinterpret_cast<float4 *>(vector)[i * blockDim.x + tid];
        // 向量乘法并累加向量内的4个结果，得到该向量内部的乘加结果
        thread_local_sum += mat4.x * vec4.x;
        thread_local_sum += mat4.y * vec4.y;
        thread_local_sum += mat4.z * vec4.z;
        thread_local_sum += mat4.w * vec4.w;
    }
    // reduce to get the final val
    // 以上仅得到了每个向量的内部乘加结果，故还需要reduce得到matrix的一行乘加vector的最终结果
    float reduce_res = blockReduce<SumOp, float>(thread_local_sum);
    // store to gmem
    if (tid == 0) {
        res[blockIdx.x] = reduce_res;
    }
    __syncthreads();
}

template<int VECS_PER_THREAD, int VEC_SIZE>
__global__ void gemv(half *matrix, half *vector, half *res, int cols) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    //float thread_local_sum = 0.0f;
    half thread_local_sum = 0;
    for (int i = 0; i < VECS_PER_THREAD; i++) {
        float4 mat4 = reinterpret_cast<float4 *>(matrix)[bid * (cols / VEC_SIZE) + i * blockDim.x + tid]; // 4 * half2
        float4 vec4 = reinterpret_cast<float4 *>(vector)[i * blockDim.x + tid];
        // 与fp32的gemv不同点在于，向量宽度由4变为8，满足128bit的CUDA线程最大读写宽度
        // 所以依然可以用float4表示读取的偏移宽度，half也OK，只是CUDA没有half8这个内置类型，需要自定义half8这个struct
        // 然后再转成half2，调用half2 intrinsic做计算
        auto *vec_h1 = (__half2 *) &vec4.x;
        auto *vec_h2 = (__half2 *) &vec4.y;
        auto *vec_h3 = (__half2 *) &vec4.z;
        auto *vec_h4 = (__half2 *) &vec4.w;
        auto *mat_h1 = (__half2 *) &mat4.x;
        auto *mat_h2 = (__half2 *) &mat4.y;
        auto *mat_h3 = (__half2 *) &mat4.z;
        auto *mat_h4 = (__half2 *) &mat4.w;
        __half2 res1 = __hmul2(*mat_h1, *vec_h1);
        __half2 res2 = __hmul2(*mat_h2, *vec_h2);
        __half2 res3 = __hmul2(*mat_h3, *vec_h3);
        __half2 res4 = __hmul2(*mat_h4, *vec_h4);
        __half2 res = __hadd2(__hadd2(__hadd2(res1, res2), res3), res4);
        thread_local_sum = __hadd(res.x, res.y);
        // float2 res1 = __half22float2(__hmul2(*mat_h1, *vec_h1));
        // float2 res2 = __half22float2(__hmul2(*mat_h2, *vec_h2));
        // float2 res3 = __half22float2(__hmul2(*mat_h3, *vec_h3));
        // float2 res4 = __half22float2(__hmul2(*mat_h4, *vec_h4));
        // thread_local_sum += res1.x;
        // thread_local_sum += res1.y;
        // thread_local_sum += res2.x;
        // thread_local_sum += res2.y;
        // thread_local_sum += res3.x;
        // thread_local_sum += res3.y;
        // thread_local_sum += res4.x;
        // thread_local_sum += res4.y;
        //if(i == 0 && tid == 0 && bid == 0) {
        //printf("thread sum = %f\n", (float)thread_local_sum); // 8
        // printf("res1.x = %f\n", res1.x); // 1
        //}
    }
    //reduce to get the final val
    // 以上仅得到了每个向量的内部乘加结果，故还需要reduce得到matrix的一行乘加vector的最终结果
    half reduce_res = blockReduce<SumOp, half>(thread_local_sum);
    //store to gmem
    if (tid == 0) {
        printf("block reduce_res = %f\n", (float) reduce_res);
        res[blockIdx.x] = reduce_res;
    }
    __syncthreads();
}

/**
 * @brief 模板类，用于启动GPU矩阵乘向量运算。
 *
 * @tparam VECS_PER_THREAD 每个线程处理的pack data数量。
 * @tparam VEC_SIZE 向量化load/store时，pack data的size。
 * @tparam THREAD_NUMS 每个block的线程数量。
 */
template<int VECS_PER_THREAD, int VEC_SIZE, int THREAD_NUMS>
struct DispatchLauncher {
    /**
     * @brief 启动GPU矩阵乘向量运算。
     *
     * 此函数用于启动一个特定配置的GPU并行计算任务，执行矩阵乘以向量的操作。
     *
     * @tparam T 数据类型，支持CUDA支持的任意数据类型。
     * @param d_mat 矩阵在GPU内存中的指针。
     * @param d_vec 向量在GPU内存中的指针。
     * @param d_dst 结果向量在GPU内存中的指针。
     * @param M 矩阵的行数。
     * @param N 矩阵的列数或向量的长度。
     */
    template<typename T>
    static void launcher(T *d_mat, T *d_vec, T *d_dst, int M, int N) {
        // 一行使用一个block
        dim3 Grid(M);
        dim3 Block(THREAD_NUMS);
        float milliseconds = 0;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        printf("calling\n");
        gemv<VECS_PER_THREAD, VEC_SIZE><<<Grid, Block>>>(d_mat, d_vec, d_dst, N);
        cudaError_t result = cudaGetLastError();
        if (result) {
            throw std::runtime_error(
                std::string("[ERROR] CUDA runtime error: ") + (cudaGetErrorEnum(result)) + " " + __FILE__ + ":" + std::to_string(__LINE__) +
                " \n");
        }
        printf("called\n");
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("gemv latency = %f ms\n", milliseconds);
    }
};

// vec * mat, mat is row major
// [1, N] * [N, M]
// logits * v
// 有关fp32/fp16 fma和add的各种重载操作
namespace gemv2 {
    // 定义一个结构体half8，用于存储8个半精度浮点数（half2类型）
    struct half8 {
        __half2 h1;
        __half2 h2;
        __half2 h3;
        __half2 h4;

        __device__ half8 &operator =(const half8 &h8) {
            h1 = h8.h1;
            h2 = h8.h2;
            h3 = h8.h3;
            h4 = h8.h4;
            return *this;
        }
    };

    /**
     * @brief 用于计算矩阵中每一行需要的线程数量。
     *
     * 此结构体模板用于确定在处理矩阵操作时，给每行矩阵应该分配多少个线程。
     * 因为是读一行矩阵元素进行处理，那么可以进行向量化读取与写入。
     * 对于fp32 是32bits -->4字节（cuda能读最大128bit -->16字节，4个fp32）。
     * 对于fp16 是16bits -->2字节，所以可以进行8个half的向量化读取与写入。
     * sizeof返回的是字节数
     *
     * @tparam M 矩阵的列数。
     * @tparam T 元素的类型。
     */
    template<int M, typename T>
    struct get_threads_per_mat_row {
        static const int value = M * sizeof(T) / 16;
    };

    inline __device__ float add(float a, float b) {
        return a + b;
    }

    inline __device__ float4 add(float4 a, float4 b) {
        float4 c;
        c.x = gemv2::add(a.x, b.x);
        c.y = gemv2::add(a.y, b.y);
        c.z = gemv2::add(a.z, b.z);
        c.w = gemv2::add(a.w, b.w);
        return c;
    }

    inline __device__ __half add(__half a, __half b) {
        return __hadd(a, b);
        //half+half is not really adding, its so weird, which  cause our result is 32, not 256
        // return (half)((float)a+(float)b);
    }

    inline __device__ __half2 add(const __half2 &a, const __half2 &b) {
        __half2 res{};
        res.x = gemv2::add(a.x, b.x);
        res.y = gemv2::add(a.y, b.y);
        return res;
    }

    inline __device__ half8 add(const half8 &a, const half8 &b) {
        half8 c{};
        c.h1 = gemv2::add(a.h1, b.h1);
        c.h2 = gemv2::add(a.h2, b.h2);
        c.h3 = gemv2::add(a.h3, b.h3);
        c.h4 = gemv2::add(a.h4, b.h4);
        return c;
    }

    inline __device__ __half fma(__half a, __half b, __half c) {
        // 有的编译器会不认识half intrinsic 例如__hmul或者__hadd，这很奇怪
        // 所以粗暴转成fp32计算再转回fp16
        return __hfma(a, b, c);
        //return __hadd(__hmul(a,b),c);
        // return __float2half((float)a * (float)b + (float)c);
    }


    inline __device__ __half2 fma(half a, const half2 &b, const half2 &c) {
        __half2 res{};
        res.x = gemv2::fma(a, b.x, c.x);
        res.y = gemv2::fma(a, b.y, c.y);
        return res;
    }

    inline __device__ half8 fma(half a, const half8 &b, const half8 &c) {
        half8 d{};
        d.h1 = gemv2::fma(a, b.h1, c.h1);
        d.h2 = gemv2::fma(a, b.h2, c.h2);
        d.h3 = gemv2::fma(a, b.h3, c.h3);
        d.h4 = gemv2::fma(a, b.h4, c.h4);
        return d;
    }

    inline __device__ float fma(float a, float b, float c) {
        return std::fma(a, b, c);
    }

    inline __device__ float4 fma(float a, float4 b, float4 c) {
        float4 d;
        d.x = gemv2::fma(a, b.x, c.x);
        d.y = gemv2::fma(a, b.y, c.y);
        d.z = gemv2::fma(a, b.z, c.z);
        d.w = gemv2::fma(a, b.w, c.w);
        return d;
    }
} // namespace gemv2


// 1个block处理一个[1, M], 循环处理完[N, M]
// for fp32: <64, M * sizeof(T) / 16 = M / 4, 4>
// [1,N] * [N,M]
// logits(QK) * V
/**
 * @brief 完成 向量 * 矩阵的GPU并行计算任务。
 *
 * @tparam THREADS_PER_BLOCK 每个block中的线程数量。
 * @tparam THREADS_PER_VALUE 对于矩阵的列数M，在packed时，需要多少个线程才能处理完。
 * @tparam VEC_SIZE pack data的大小。// 对于fp32是32bits -->4字节（cuda能读最大128bit -->16字节，4个fp32）。
 *                                 // 对于fp16是16bits -->2字节，所以可以进行8个half的向量化读取与写入。
 *
 * @tparam T 数据类型。
 *
 *
 * @param matrix 矩阵。
 * @param vector 向量。
 * @param res 结果。
 * @param N 矩阵的行数。
 * @param M 矩阵的列数。
 *
 * */
template<int THREADS_PER_BLOCK, int THREADS_PER_VALUE, int VEC_SIZE, typename T>
__global__ void gemv2_kernel_template(T *matrix, T *vector, T *res, int N, int M) {
    // 在THREADS_PER_BLOCK = 256，[N, M] --> N = 256, M = 512时.使用float
    // VEC_SIZE = 4
    // THREADS_PER_VALUE = 128

    /*根据编译期常量获取每个thread处理的行列号*/
    // 前线程的索引
    int tid = threadIdx.x;
    /**
     * mat_i 计算当前线程负责处理的矩阵行的索引。
     * 假设 THREADS_PER_VALUE 为 128，THREADS_PER_BLOCK 为 256，那么每个矩阵行会分配给 2 个线程来处理。
     * tid / THREADS_PER_VALUE 确保前 2 个线程处理第 0 行,依此类推。
     */
    int mat_i = tid / THREADS_PER_VALUE;
    /**
     * mat_j 计算当前线程在矩阵行中处理的具体列位置.
     * tid % THREADS_PER_VALUE 计算当前线程在其分配的矩阵行中处理的位置索引。
     * 乘以 VEC_SIZE 是因为每个线程处理多个连续的值（VEC_SIZE 表示每个线程处理的值的大小）。
     *
     * */
    int mat_j = tid % THREADS_PER_VALUE * VEC_SIZE;
    /**
     * 一个block在一次迭代中处理多少行。
     * THREADS_PER_VALUE表示对于矩阵的列数M，在packed时，需要多少个线程才能处理完。
     * */
    constexpr int ROW_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_VALUE;
    //
    __shared__ T out_smem[512];

    // 使用泛型类型来表示out
    using VecType = typename std::conditional<std::is_same<T, __half>::value, gemv2::half8, float4>::type;
    VecType out;
    // inter-block 累加
    // 在一个block内将ROW_PER_ITER那么多行累加。
    for (int ti = mat_i; ti < N; ti += ROW_PER_ITER) {
        VecType mat = *reinterpret_cast<VecType *>(&matrix[ti * M + mat_j]);
        T logits = vector[ti];
        out = gemv2::fma(logits, mat, out);
    }
    // intra-block 累加
    for (int ROWS_PER_BLOCK = ROW_PER_ITER; ROWS_PER_BLOCK >= 2; ROWS_PER_BLOCK >>= 1) {
        int midpoint = ROWS_PER_BLOCK / 2;
        if (mat_i >= midpoint && mat_i < ROWS_PER_BLOCK) {
            *reinterpret_cast<VecType *>(&out_smem[(mat_i - midpoint) * M + mat_j]) = out;
        }
        __syncthreads();
        if (mat_i < midpoint) {
            // ROW_PER_ITER中上半部分out和下半部分out相加
            out = gemv2::add(*reinterpret_cast<VecType *>(&out_smem[mat_i * M + mat_j]), out);
        }
        __syncthreads();
    }
    if (mat_i == 0) {
        *reinterpret_cast<VecType *>(&res[mat_j]) = out;
    }
}

template<int THREADS_PER_BLOCK, int THREADS_PER_VALUE, int VEC_SIZE>
struct DispatchLauncher2 {
    template<typename T>
    static void launcher(T *d_mat, T *d_vec, T *d_dst, int M, int N) {
        dim3 Grid(1);
        dim3 Block(THREADS_PER_BLOCK);
        float milliseconds = 0;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        printf("calling\n");
        // 启动cuda kernel
        // 打印模板参数

        printf("THREADS_PER_BLOCK = %d, THREADS_PER_VALUE = %d, VEC_SIZE = %d\n", THREADS_PER_BLOCK, THREADS_PER_VALUE, VEC_SIZE);
        gemv2_kernel_template<THREADS_PER_BLOCK, THREADS_PER_VALUE, VEC_SIZE, T><<<Grid, Block>>>(d_mat, d_vec, d_dst, N, M);
        cudaError_t result = cudaGetLastError();
        if (result) {
            throw std::runtime_error(
                std::string("[ERROR] CUDA runtime error: ") + (cudaGetErrorEnum(result)) + " " + __FILE__ + ":" + std::to_string(__LINE__) +
                " \n");
        }
        printf("called\n");
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("gemv latency = %f ms\n", milliseconds);
    }
};
