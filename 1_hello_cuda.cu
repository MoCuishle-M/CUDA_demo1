#include <algorithm>   // std::fill_n
#include <iostream>    // snprintf
#include <math.h>      // expf, log
#include <stdlib.h>    // rand
#include <string>      // std::string
#include <vector>      // std::vector
#include <stdio.h>
#include <fstream>
#include <cublas_v2.h>
#include <cuda_runtime.h>


// 错误检查宏
#define CUDA_CHECK_ERROR(err) { if (err != cudaSuccess) { std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; exit(EXIT_FAILURE); } }
#define CUBLAS_CHECK_ERROR(err) { if (err != CUBLAS_STATUS_SUCCESS) { std::cerr << "CUBLAS Error: " << _cublasGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; exit(EXIT_FAILURE); } }

// CUBLAS错误字符串
const char* _cublasGetErrorString(cublasStatus_t error) {
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:          return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:  return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:     return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:    return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:    return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:    return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:   return "CUBLAS_STATUS_INTERNAL_ERROR";
        default: return "Unknown CUBLAS error";
    }
}

void CPUlinear(const float *input, const float *weight, float *output,
               int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int l = 0; l < k; l++) {
                output[i * n + j] += input[i * k + l] * weight[l * n + j];
            }
        }
    }
}

bool CheckResult(const float *CPUoutput, const float *GPUoutput, const int output_size) {
    for (int i = 0; i < output_size; i++) {
        if (i < 5) {
            printf("0th res, CPUoutput = %f, GPUoutput = %f\n", CPUoutput[i], GPUoutput[i]);
        }
        if (fabs(CPUoutput[i] - GPUoutput[i]) > 1e-6) {
            printf("the %dth res is wrong, CPUoutput = %f, GPUoutput = %f\n", i, CPUoutput[i], GPUoutput[i]);
            return false;
        }

    }
    return true;
}

int main(int argc, char *argv[]) {
    // const int seqlen = 13;
    // const int hidden_units = 4096;
    // const int vocab_size = 32;
    // const int inter_size = 10;
    // int hidden_units_2 = 0;
    // int output_size = 0;
    //
    // hidden_units_2 = hidden_units * hidden_units;
    // output_size = seqlen * hidden_units;
    // // debug info, better to retain: std::cout <<"batch_size=" << batch_size << "  vocab_size=" << vocab_size << std::endl;
    // float *h_w;
    // float *d_w;
    // h_w = (float *) malloc(sizeof(float) * hidden_units_2);
    // cudaMalloc((void **) &d_w, sizeof(float) * hidden_units_2);
    // for (int i = 0; i < hidden_units_2; i++) {
    //     h_w[i] = (float) (i % 3); // 1 2 1 2
    // }
    //
    // float *h_in = (float *) malloc(sizeof(float) * hidden_units * seqlen);
    // float *d_in;
    // cudaMalloc((void **) &d_in, sizeof(float) * seqlen * hidden_units);
    // for (int i = 0; i < hidden_units * seqlen; i++) {
    //     h_in[i] = (float) (i % 3);
    // }
    //
    // float *h_out = (float *) malloc(sizeof(float) * output_size);
    // float *d_out;
    // cudaMalloc((void **) &d_out, sizeof(float) * output_size);
    // CHECK(cudaMemcpy(d_in, h_in, sizeof(float) * hidden_units * seqlen, cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpy(d_w, h_w, sizeof(float) * hidden_units_2, cudaMemcpyHostToDevice));
    // DataType type = getTensorType<float>();
    // WeightType wtype = getWeightType<float>();
    // TensorWrapper<float> *in = new TensorWrapper<float>(Device::GPU, type, {seqlen, hidden_units}, d_in);
    // BaseWeight<float> weight;
    // weight.shape = {hidden_units, hidden_units};
    // weight.data = d_w;
    // weight.type = wtype;
    // TensorWrapper<float> *out;
    // out = new TensorWrapper<float>(Device::GPU, type, {seqlen, hidden_units}, d_out);
    //
    // cublasHandle_t cublas_handle;
    // cublasLtHandle_t cublaslt_handle;
    // cublasCreate(&cublas_handle);
    // cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);
    // cublasWrapper *cublas_wrapper = new cublasWrapper(cublas_handle, cublaslt_handle);
    // cublas_wrapper->setFP32GemmConfig();
    // // // debug info, better to retain:
    // // std::cout << "before launch kernel" << std::endl;
    // // launchLinearGemm(in, weight, out, cublas_wrapper);
    // // // debug info, better to retain:
    // // std::cout << "after launch kernel" << std::endl;
    // // // debug info, better to retain:
    // // std::cout << "cuda memcpy device to host" << std::endl;
    // // // Note: remember to memcpy from device to host and define the correct copy size(mul the sizeof(dtype)), or will cause segment fault
    // // CHECK(cudaMemcpy(h_out, d_out, sizeof(float) * output_size, cudaMemcpyDeviceToHost));
    // float *CPUout = (float *) malloc(sizeof(float) * output_size);
    // CPUlinear(h_in, h_w, CPUout, seqlen, hidden_units, hidden_units);
    //
    // // bool is_right = CheckResult(CPUout, h_out, output_size);
    // // // debug info, better to retain:
    // // std::cout << "before free" << std::endl;
    // // std::cout << "linear passed" << std::endl;
    // free(h_in);
    // free(h_w);
    // free(h_out);
    // free(CPUout);
    // cudaFree(d_in);
    // cudaFree(d_w);
    // cudaFree(d_out);
    printf("--------------------------------------end--------------------------------------\n");
    const int M = 2, N = 2, K = 2;
    float h_A[M*K] = {1, 2, 3, 4};
    float h_B[K*N] = {5, 6, 7, 8};
    float h_C[M*N] = {0};

    float *d_A, *d_B, *d_C;
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;

    // 分配设备内存
    cudaStat = cudaMalloc((void**)&d_A, M * K * sizeof(float));
    CUDA_CHECK_ERROR(cudaStat);

    cudaStat = cudaMalloc((void**)&d_B, K * N * sizeof(float));
    CUDA_CHECK_ERROR(cudaStat);

    cudaStat = cudaMalloc((void**)&d_C, M * N * sizeof(float));
    CUDA_CHECK_ERROR(cudaStat);

    // 创建CUBLAS句柄
    stat = cublasCreate(&handle);
    CUBLAS_CHECK_ERROR(stat);

    // 复制数据到设备
    stat = cublasSetMatrix(M, K, sizeof(float), h_A, M, d_A, M);
    CUBLAS_CHECK_ERROR(stat);

    stat = cublasSetMatrix(K, N, sizeof(float), h_B, K, d_B, K);
    CUBLAS_CHECK_ERROR(stat);

    // 执行GEMM操作: C = alpha*A*B + beta*C
    const float alpha = 1.0f;
    const float beta = 0.0f;
    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);
    CUBLAS_CHECK_ERROR(stat);

    // 复制结果回主机
    stat = cublasGetMatrix(M, N, sizeof(float), d_C, M, h_C, M);
    CUBLAS_CHECK_ERROR(stat);

    // 打印结果
    std::cout << "Result matrix C:" << std::endl;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << h_C[i + j * M] << " ";
        }
        std::cout << std::endl;
    }

    // 清理资源
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}