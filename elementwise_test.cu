#include "elementwise_test.cuh"

#include <device_launch_parameters.h>
#include <cuda_fp16.h>

//template<typename T>
//struct GeluFunctor {
//  __device__ T operator()(const T x) const {
//	return static_cast<T>(0.5) * x * (static_cast<T>(1.0) + erf(x / sqrt(static_cast<T>(2.0))));
//  }
//};
//
//// Specialization for __half
//template<>
//struct GeluFunctor<__half> {
//  __device__ __half operator()(const __half x) const {
//	float xf = __half2float(x);
//	float result = 0.5f * xf * (1.0f + erff(xf / sqrtf(2.0f)));
//	return __float2half(result);
//  }
//};
//
//template<typename T>
//struct GeluFunctorFactory {
//  __device__ GeluFunctor<T> operator()() const {
//	return GeluFunctor<T>();
//  }
//};
__device__ float TanhApprox(float x) {
  // ptx指令，是CUDA的更底层的语言，类似于汇编对于C/C++
  //float r;
  //asm("tanh.approx.f32 %0,%1; \n\t" : "=f"(r) : "f"(x));
  //return r;
  return tanhf(x); // CUDA内置的math API
}

// gelu公式：x / 2 * (1 + tan(0.7978845608028654 * (x + 0.044714998453855515 * x^3)))
template<typename T>
struct GeluFunctor {
  static constexpr T alpha = static_cast<T>(0.7978845608028654);
  static constexpr T beta = static_cast<T>(0.044714998453855515);

  __device__  __forceinline__ GeluFunctor() = default;

  __device__  __forceinline__ T operator()(T x) const {
	const T half = static_cast<T>(0.5);
	const T one = static_cast<T>(1);
	const T tanh_in = alpha * (x + beta * x * x * x);
	return half * x * (one + tanh(tanh_in));
  }
};

template<>
struct GeluFunctor<__half> {
  // 偷了个懒，直接把L26和L27拿过来用
  static constexpr float alpha = GeluFunctor<float>::alpha;
  static constexpr float beta = GeluFunctor<float>::beta;

  __device__  __forceinline__ __half operator()(const __half x) const {
	// Note: when you have ampere GPU, you can enable the line45-50 method to get performance improvement by half intrinsic instead of static_cast half to fp32.
	const float tanh_in =
		__half2float(__float2half_rn(alpha) * (x + __float2half_rn(beta) * x * x * x));
	const float tanh_out = TanhApprox(tanh_in);
	return __float2half_rn(0.5f) * x * (__float2half_rn(1.0f) + __float2half_rn(tanh_out));
	// Note: half to float will lose performance using static_cast, because static_cast will be compiled to more instructions than half intrinsic,
	// so you should better use half intrinsic when you have ampere GPU, you can enable 44-47 line
	// return static_cast<half>(float_functor(static_cast<float>(x)));
  }
  // Note: when you have ampere GPU, you can enable the "Apply2" method to get performance improvement by half2 intrinsic.
  static __device__  __forceinline__ void Apply2(__half* y, const __half* x) {
	const half2 x2 = *(reinterpret_cast<const half2*>(x)); // L89行已经求出了offset，这里直接把指针转换为向量类型并解引用即可得到向量数据
	const float2 tanh_in = __half22float2(
		__hmul2(__float2half2_rn(alpha),
				__hadd2(x2, __hmul2(__hmul2(__hmul2(__float2half2_rn(beta), x2), x2), x2))));
	float2 tanh_out;
	tanh_out.x = TanhApprox(tanh_in.x); // tanh之所以转为fp32类型计算，是因为NV GPU貌似不支持tanh的half intrinsic，理想状况下，当然是希望所有计算都是half2一把梭
	tanh_out.y = TanhApprox(tanh_in.y);
	const half2 y2 = __hmul2(__hmul2(__float2half2_rn(0.5F), x2),
							 __hadd2(__float2half2_rn(1.0F), __float22half2_rn(tanh_out)));
	*reinterpret_cast<half2*>(y) = y2; // 向量化写回结果到显存
  }
};

template<typename T>
struct GeluFunctorFactory {
  __device__ GeluFunctor<T> operator()() const {
	return GeluFunctor<T>();
  }
};

int main() {
  int n = 1000;

  auto *h_input = new __half[n];
  auto *h_output = new __half[n];
  for (int i = 0; i < n; i++)
  {
	h_input[i] = (__half)(i);
  }
  __half * d_input, *d_output;
  cudaMalloc((void **)&d_input, n * sizeof(__half));
  cudaMalloc((void **)&d_output, n * sizeof(__half));
  cudaMemcpy(d_input, h_input, sizeof(__half) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_output, h_output, sizeof(__half) * n, cudaMemcpyHostToDevice);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  GeluFunctorFactory<__half> gelu_functor;
  cudaError_t status = oneflow::cuda::elementwise::UnaryWithFactory(gelu_functor, n, d_output, d_input, stream);

  if (status != cudaSuccess) {
	// 处理错误
	printf("CUDA error: %s\n", cudaGetErrorString(status));
  }
  cudaMemcpy(h_output, d_output, sizeof(__half) * n, cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++)
  {
	printf("%f\n", __half2float(h_output[i]));
  }
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
  cudaFree(d_input);
  cudaFree(d_output);
  delete[] h_input;
  h_input = nullptr;
  delete[] h_output;
  h_output = nullptr;

  return 0;
}