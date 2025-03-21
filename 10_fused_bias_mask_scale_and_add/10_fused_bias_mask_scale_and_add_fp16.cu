#include <cuda.h>
#include <cuda_fp16.h>
#include "cuda_runtime.h"
typedef __half half;
typedef __half2 half2;



template<typename T>
struct MaskScaleAndElementwiseAddFunctor {
  MaskScaleAndElementwiseAddFunctor(const uint8_t* mask, const T* add_val, float scale)
      : mask(mask), add_val(add_val), scale(scale) {}
  __device__ T Compute(T x, int64_t i) const {
    // mask和scale先做计算，然后结果再和x做计算，最后element wise相加
    return x * static_cast<T>(static_cast<bool>(mask[i]) * scale) + add_val[i];
  }
  const uint8_t* mask;
  const T* add_val;
  float scale;
};

template<>
struct MaskScaleAndElementwiseAddFunctor<half> {
  MaskScaleAndElementwiseAddFunctor(const uint8_t* mask, const half* add_val, float scale)
      : mask(mask), add_val(add_val), scale(scale) {}
  // half标量版本的MaskScaleAndElementwiseAdd，与L15区别不大，注意: 有的GPU在有的nvcc和cuda版本下，没有重载half*half的直接相乘版本，此时需要用hmul(half,half)来代替或者两个half强转为fp32来相乘再转回half,比如(half)((float)x * (float)y)
  __device__ half Compute(half x, int64_t i) const {
    return x * static_cast<half>(static_cast<bool>(mask[i]) * scale) + add_val[i];
  }
  // half向量版本的MaskScaleAndElementwiseAdd，不仅支持L32和L33所示的向量化读取，也支持L39所示的向量化计算，这与fp32的向量化是不同的.
  __device__ half2 ComputeHalf2(half2 x, int64_t i) const {
    const char2* mask_c2 = reinterpret_cast<const char2*>(mask);
    const half2* add_val_h2 = reinterpret_cast<const half2*>(add_val);
    char2 mask_val = mask_c2[i]; // 向量化读取
    half2 one_or_zero_h2; // 向量化读取
    half2 h2_scale = __float2half2_rn(scale); // float->half2, ep. 1.0 => (1.0, 1.0)
    one_or_zero_h2.x = mask_val.x;
    one_or_zero_h2.y = mask_val.y;
    return __hadd2(__hmul2(__hmul2(x, one_or_zero_h2), h2_scale), add_val_h2[i]);
  }
  const uint8_t* mask;
  const half* add_val;
  float scale;
};

// biasAdd的输入两个，x.shape={rows, cols}, bias.shape={cols}, 所以需要在L59通过除余循环读取这cols个bias
template<typename FUNCTOR>
__global__ void FusedBiasAddCUDAKernelHalf2(FUNCTOR functor, const int elem_cnt,
                                            const int bias_size, const half* x, const half* bias,
                                            half* y) {
  const int h2_elem_cnt = elem_cnt / 2; // 读取的粒度由half变成了half2，那自然元素数量就少了一半
  const int h2_bias_size = bias_size / 2;
  const auto* x_h2 = reinterpret_cast<const half2*>(x); // 强转为向量指针后在L58读取
  const auto* bias_h2 = reinterpret_cast<const half2*>(bias);
  auto* y_h2 = reinterpret_cast<half2*>(y);
  // 保证有限线程数处理完所有数据
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < h2_elem_cnt;
       i += blockDim.x * gridDim.x){
    /*
     * __host__ __device__ __half2 __hadd2(const __half2 a, const __half2 b)
     *
     *  return half2 ▶ The sum of vectors a and b.
     */
    half2 x_i = __hadd2(x_h2[i], bias_h2[i % h2_bias_size]); //
    y_h2[i] = functor.ComputeHalf2(x_i, i);
  }
}

int main(){
  constexpr int ele_cnt = 1000;
  float scale = 0.5;
  auto *h_mask_tensor = new uint8_t[ele_cnt];
  auto *h_add_val = new __half[ele_cnt];
  for (int i = 0; i < ele_cnt; i++){
    h_mask_tensor[i] = (uint8_t)(i);
    h_add_val[i] = (float)(i);
  }
  int bias_size = 10;

  auto *h_input = (__half*) malloc(sizeof(__half) * ele_cnt);
  auto *h_output = (__half*) malloc(sizeof(__half) * ele_cnt);
  auto *h_bias = (__half*) malloc(sizeof(__half) * bias_size);
  for (int i = 0; i < ele_cnt; i++)
  {
    h_input[i] = (__half)(i);
  }
  __half *d_input, *d_output, *d_bias;
  cudaMalloc((void **)&d_input, ele_cnt * sizeof(__half));
  cudaMalloc((void **)&d_output, ele_cnt * sizeof(__half));
  cudaMalloc((void **)&d_bias, bias_size * sizeof(__half));
  cudaMemcpy(d_input, h_input, sizeof(__half) * ele_cnt, cudaMemcpyHostToDevice);
  cudaMemcpy(d_output, h_output, sizeof(__half) * ele_cnt, cudaMemcpyHostToDevice);
  cudaMemcpy(d_bias, h_bias, sizeof(__half) * bias_size, cudaMemcpyHostToDevice);
  uint8_t *d_mask_tensor;
  __half *d_add_val;
  cudaMalloc((void **)&d_mask_tensor, ele_cnt * sizeof(uint8_t));
  cudaMalloc((void **)&d_add_val, ele_cnt * sizeof(__half));
  cudaMemcpy(d_mask_tensor, h_mask_tensor, sizeof(uint8_t) * ele_cnt, cudaMemcpyHostToDevice);
  cudaMemcpy(d_add_val, h_add_val, sizeof(__half) * ele_cnt, cudaMemcpyHostToDevice);

  cudaDeviceProp deviceProp{};
  cudaGetDeviceProperties(&deviceProp, 0);
  int maxblocks = deviceProp.maxGridSize[0];
  int blockSize = 256;
  int gridSize = std::min((ele_cnt + blockSize - 1) / blockSize, maxblocks);
  MaskScaleAndElementwiseAddFunctor<half> mask_scale_elementwise_add_func(
      h_mask_tensor, h_add_val, scale);
  FusedBiasAddCUDAKernelHalf2<<<gridSize ,blockSize>>>(mask_scale_elementwise_add_func, ele_cnt, bias_size, d_input, d_bias,
      d_output);
  cudaMemcpy(h_output, d_output, sizeof(__half) * ele_cnt, cudaMemcpyDeviceToHost);

  free(h_input);
  free(h_output);
  free(h_bias);
  delete[] h_add_val;
  h_add_val = nullptr;
  delete[] h_mask_tensor;
  h_mask_tensor = nullptr;
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_bias);
  cudaFree(d_mask_tensor);
  cudaFree(d_add_val);
}