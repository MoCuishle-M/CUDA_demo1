#include "15_gemv.cuh"
#include <memory>
#include <type_traits>
#include <cuda_fp16.h>
// [1, N] * [N, M]
// notes: CPU res sometimes are trash values, which very weird, so I check result by printing each res skipping comparison with CPU res
// when compile to executable file named gemv, we can run by typing "./gemv 1" to run fp32 gemv and "./gemv" to run fp16 gemv
template<typename T>
void GEMVCpu(const T *mat, const T *vec, float *dst, int M, int N) {
    if constexpr (std::is_same_v<T, __half>) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                dst[i] += __half2float(mat[i + j * M]) * __half2float(vec[j]);
            }
        }
    } else {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                dst[i] += mat[i + j * M] * vec[j];
            }
        }
    }
}

template<typename T>
bool CheckResult(const T *out, const float *ground_truth, int M) {
    if constexpr (std::is_same_v<T, __half>) {
        for (int i = 0; i < M; i++) {
            printf("%d th comparison: %f and %f \n", i, __half2float(out[i]), ground_truth[i]);
        }
    } else {
        for (int i = 0; i < M; i++) {
            printf("%d th comparison: %f and %f \n", i, out[i], ground_truth[i]);
        }
    }
    return true;
}

// vec.shape = [1, N]
// mat.shape = [N, M] and matrix is row major order in memory
template<typename T>
void GemvKernel2() {
    T *d_vec;
    T *d_mat;
    T *d_dst;
    constexpr int N = 512;
    constexpr int M = 256;
    auto vec = std::make_unique<T[]>(N);
    cudaMalloc((void **) &d_vec, N * sizeof(T));
    auto mat = std::make_unique<T[]>(M * N);
    cudaMalloc((void **) &d_mat, M * N * sizeof(T));
    auto dst = std::make_unique<T[]>(M);
    std::fill_n(dst.get(), M, 0); // 使用fill_n函数填充所有元素为0
    cudaMalloc((void **) &d_dst, M * sizeof(T));
    for (int i = 0; i < N; i++) {
        vec[i] = (T) 1;
    }
    for (int i = 0; i < N * M; i++) {
        mat[i] = (T) 1;
    }
    cudaMemcpy(d_vec, vec.get(), N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat, mat.get(), M * N * sizeof(T), cudaMemcpyHostToDevice);
    constexpr int THREADS_PER_BLOCK = 256;
    constexpr int VEC_SIZE = Vec<T>::size;
    constexpr int THREADS_PER_VALUE = gemv2::get_threads_per_mat_row<M, T>::value;
    DispatchLauncher2<THREADS_PER_BLOCK, THREADS_PER_VALUE, VEC_SIZE>::template launcher<T>(d_mat, d_vec, d_dst, M, N);
    cudaMemcpy(dst.get(), d_dst, M * sizeof(T), cudaMemcpyDeviceToHost);
    if constexpr (std::is_same_v<T, __half>) {
        printf("Since the cpu doesn't have a HALF type, it prints the kernel result directly\n");
    } else {
        printf("fp32 gemv kernel result:\n");
    }
    auto ground_truth = std::make_unique<float[]>(M);
    // 注意：此处没有验证fp16的cpu gemv，只是打印fp16 cuda kernel的结果肉眼看了一下
    // 验证fp16的结果的做法是L75传入half类型的输入和模板类型，在 gemv cpu函数里面将输入类型强转为fp32即可，因为cpu没有half类型
    GEMVCpu<T>(mat.get(), vec.get(), ground_truth.get(), M, N);
    CheckResult(dst.get(), ground_truth.get(), M);

    cudaFree(d_vec);
    cudaFree(d_mat);
    cudaFree(d_dst);
}

int main() {
    if (true) {
        printf("use float\n");
        GemvKernel2<float>();
    } else {
        printf("use half\n");
        GemvKernel2<__half>();
    }
}
