#include "15_gemv.cuh"
#include <memory>
#include <type_traits>


template<typename T>
void GEMVCpu(const T *mat, const T *vec, float *dst, int M, int N) {
    if constexpr (std::is_same_v<T, __half>) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                dst[i] += __half2float(mat[i * N + j]) * __half2float(vec[j]);
            }
        }
    } else {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                dst[i] += mat[i * N + j] * vec[j];
            }
        }
    }
}

template<typename T>
bool CheckResult(const T *out, const float *ground_truth, int M) {
    if constexpr (std::is_same_v<T, __half>) {
        for (int i = 0; i < M; i++) {
            printf("%d th comparison: %f and %f \n", i, (float) out[i], ground_truth[i]);
        }
    } else {
        for (int i = 0; i < M; i++) {
            printf("%d th comparison: %f and %f \n", i, (float) out[i], ground_truth[i]);
        }
    }
    return true;
}


template<typename T>
void GemvKernel2() {

    constexpr int N = 2048; //256 * 8
    constexpr int M = 256;

    //    initialize<T>(vec, d_vec, mat, d_mat, dst, d_dst, M, N);
    auto vec = std::make_unique<T[]>(N);
    auto mat = std::make_unique<T[]>(M * N);
    auto dst = std::make_unique<T[]>(M);
    std::fill_n(dst.get(), M, 0); // 使用fill_n函数填充所有元素为0
    T *d_vec;
    T *d_mat;
    T *d_dst;
    cudaMalloc((void **) &d_vec, N * sizeof(T));
    cudaMalloc((void **) &d_mat, M * N * sizeof(T));
    cudaMalloc((void **) &d_dst, M * sizeof(T));

    for (int i = 0; i < N; i++) {
        vec[i] = (T) 1;
    }
    for (int i = 0; i < N * M; i++) {
        mat[i] = (T) 1;
    }

    // GEMVCpu(mat, vec, dst, M, N);

    cudaMemcpy(d_vec, vec.get(), N * sizeof(T), cudaMemcpyHostToDevice);

    cudaMemcpy(d_mat, mat.get(), M * N * sizeof(T), cudaMemcpyHostToDevice);
    constexpr int THREAD_NUMS = 256;
    constexpr int VEC_SIZE = Vec<T>::size;
    constexpr int VECS_PER_THREAD = (N / THREAD_NUMS) / VEC_SIZE; // 1 for half, 2 for fp32
    DispatchLauncher<VECS_PER_THREAD, VEC_SIZE, THREAD_NUMS>::template launcher<T>(d_mat, d_vec, d_dst, M, N);

    CHECK(cudaMemcpy(dst.get(), d_dst, M * sizeof(T), cudaMemcpyDeviceToHost));
    if constexpr (std::is_same_v<T, __half>) {
        printf("Since the cpu doesn't have a HALF type, it prints the kernel result directly\n");
    } else {
        printf("fp32 gemv kernel result:\n");
    }
    auto ground_truth = std::make_unique<float[]>(M);
    // GEMVCpu(mat, vec, ground_truth, M, N);
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
        GemvKernel2<float>();
    } else {
        GemvKernel2<__half>();
    }
}
