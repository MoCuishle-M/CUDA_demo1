#include <cuda.h>
#include <cstdio>
#include "cuda_runtime.h"

// 实现fp32的fused biasadd mask scale and add的融合算子
// biasadd + mask + scale + elemwise_add四个算子的融合
// （x + bias） * mask * scale + addend;

template<typename T>
struct MaskScaleAndElemwiseAddFunctor
{
    // 有参构造函数
    MaskScaleAndElemwiseAddFunctor(const uint8_t * mask, const T * add_val, float scale)
    :_mask(mask), _add_val(add_val), _scale(scale)
    {}

    // 重载运算符（）
    __device__ T operator()(T x, int i) const
    {
        return x * static_cast<T>(static_cast<bool>(_mask[i]) * _scale) + _add_val[i];
    }

    const uint8_t * _mask;
    const T * _add_val;
    float _scale;
};

template<int biasSize, typename FUNCTOR, typename T>
__global__ void FusedBaisAdd(FUNCTOR functor, T * dx, T * dy, T * d_bias, const int n, const int bias_size)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = gid; i < n; i += gridDim.x * blockDim.x)
    {
        T tmp = dx[i] + d_bias[i % bias_size];
        dy[i] = functor(tmp, i);
    }
}

// 使用向量化进行存取
template<int biasSize, typename FUNCTOR, typename T>
__global__ void FusedBaisAddVecSmem(FUNCTOR functor, T * dx, T * dy, T * d_bias, const int n, const int bias_size){
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    __shared__ T smem[biasSize];

    // 将d_bias放在shared memory上
    if (tid < bias_size)
        smem[tid] = d_bias[tid];
    __syncthreads();

    for (int i = gid; i < n / 4; i += gridDim.x * blockDim.x){
        float4 a = reinterpret_cast<float4 *>(dx)[i];
        float4 b;

        b.x = functor(a.x + smem[(i * 4) % bias_size], i * 4);
        b.y = functor(a.y + smem[(i * 4 + 1) % bias_size], i * 4 + 1);
        b.z = functor(a.z + smem[(i * 4 + 2) % bias_size], i * 4 + 2);
        b.w = functor(a.w + smem[(i * 4 + 3) % bias_size], i * 4 + 3);

        reinterpret_cast<float4*>(dy)[i] = b;
    }
}

bool CheckRight(float * y, float *groundTruth, const int n)
{
    for (int i = 0; i < n; ++i)
    {
        if (y[i] != groundTruth[i])
        {
            printf("y[%d] : %f \n", i, y[i]);
            printf("groundTruth[%d] : %f\n", i, groundTruth[i]);
            return false;
        }
    }
    return true;
}

int main()
{
    constexpr int n = 100000000;
    constexpr int bias_size = 10;
    
    float scale = 0.5;
    auto *h_mask_tensor = new uint8_t[n];
    auto *h_add_val = new float[n];
    // 初始化
    for (int i = 0; i < n; ++i){
      h_mask_tensor[i] = (uint8_t)(i);
      h_add_val[i] = (float)(i);
    }

    auto *h_input = (float *)malloc(sizeof(float) * n);
    auto *h_output = (float *)malloc(sizeof(float) * n);
    auto *h_bias = (float *)malloc(sizeof(float) * bias_size);
    for (int i = 0; i < n; ++i)
    {
      h_input[i] = (float)(i);
      h_output[i] = 0.0f;
    }
    for (int i = 0; i < bias_size; ++i)
      h_bias[i] = static_cast<float >(i);

    auto *groundTruth = (float *)malloc(sizeof(float) * n);
    for (int i = 0; i < n; ++i){
      groundTruth[i] = (h_input[i] + h_bias[i % bias_size]) * static_cast<float>(static_cast<bool>(h_mask_tensor[i]) * scale) +
          h_add_val[i];
    }

    float *d_input, *d_output, * d_bias;
    cudaMalloc((void **)&d_input, sizeof(float) * n);
    cudaMalloc((void **)&d_output, sizeof(float) * n);
    cudaMalloc((void **)&d_bias, sizeof(float) * bias_size);
    cudaMemcpy(d_input, h_input, sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, sizeof(float) * bias_size, cudaMemcpyHostToDevice);
    uint8_t * d_mask_tensor;
    float * d_add_val;
    cudaMalloc((void **)&d_mask_tensor, sizeof(uint8_t) * n);
    cudaMalloc((void **)&d_add_val, sizeof(float) * n);
    cudaMemcpy(d_mask_tensor, h_mask_tensor, sizeof(uint8_t) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_add_val, h_add_val, sizeof(float) * n, cudaMemcpyHostToDevice);

    cudaDeviceProp deviceProp{};
    cudaGetDeviceProperties(&deviceProp, 0);
    int blockSize = 512;
    int gridSize = std::min((n + blockSize - 1) / blockSize, deviceProp.maxGridSize[0]);

    MaskScaleAndElemwiseAddFunctor<float> functor(d_mask_tensor, d_add_val, scale);

    dim3 Block(blockSize);
    dim3 Grid(gridSize);

    float milliseconds = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    

    FusedBaisAdd<bias_size><<<Grid, Block>>>(functor, d_input, d_output, d_bias, n, bias_size);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_output, d_output, sizeof(float) * n, cudaMemcpyDeviceToHost);

    bool isRight = CheckRight(h_output, groundTruth, n);
    if (isRight)
        printf("结果正确\n");
    else
        printf("结果错误\n");    

    printf("it costs %f s \n", milliseconds/1000);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_bias);
    cudaFree(d_add_val);
    cudaFree(d_mask_tensor);
    free(h_input);
    free(h_output);
    free(h_bias);
    free(groundTruth);
    delete[] h_mask_tensor;
    h_mask_tensor = nullptr;
    delete[] h_add_val;
    h_add_val = nullptr;
}