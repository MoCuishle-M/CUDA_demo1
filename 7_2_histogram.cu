#include <cstdio>
#include <cuda.h>
#include "cuda_runtime.h"
#include <algorithm>
#include <random>

template <int blockSize>
__global__ void histogram(const int *hist_data, int *bin_data, int N) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // 初始化共享内存中的私有直方图
  extern __shared__ unsigned int histo_s[];
  for (unsigned int binIdx = threadIdx.x; binIdx < blockSize; binIdx += blockDim.x) {
	histo_s[binIdx] = 0u;
  }
  __syncthreads();

  // 计算局部直方图
  unsigned int prev_index = -1;
  unsigned int accumulator = 0;
  unsigned int curr_index;
  for (unsigned int i = tid; i < N; i += blockDim.x * gridDim.x) {
	unsigned char value = hist_data[i];
	curr_index = value;
	if (prev_index != curr_index) {
	  if (accumulator > 0)
		atomicAdd(&(histo_s[prev_index]), accumulator);
	  prev_index = curr_index;
	  accumulator = 1;
	} else {
	  accumulator++;
	}
  }

  if (accumulator > 0)
	atomicAdd(&(histo_s[prev_index]), accumulator);

  __syncthreads();

  // 合并局部直方图到全局内存
  for (unsigned int i = threadIdx.x; i < blockSize; i += blockDim.x) {
	atomicAdd(&(bin_data[i]), histo_s[i]);
  }
}

bool CheckResult(int *out, int* groudtruth, int N){
  for (int i = 0; i < N; i++){
	if (out[i] != groudtruth[i]) {
	  printf("in checkres, out[i]=%d, gt[i]=%d\n", out[i], groudtruth[i]);
	  return false;
	}
  }
  return true;
}

int main(){
  float milliseconds = 0;
  const int N = 25600000;
  int *h_hist = (int *)malloc(N * sizeof(int));
  int *bin = (int *)malloc(256 * sizeof(int));
  int *bin_data;
  int *hist_data;
  cudaMalloc((void **)&bin_data, 256 * sizeof(int));
  cudaMalloc((void **)&hist_data, N * sizeof(int));

  for(int i = 0; i < N; i++){
	h_hist[i] = i % 256;
  }
  unsigned seed = 37; // 你可以选择任何整数值作为种子
  std::mt19937 g(seed);
  std::shuffle(h_hist, h_hist+N, g);

  int *groudtruth = (int *)malloc(256 * sizeof(int));;
  for(int j = 0; j < 256; j++){
	groudtruth[j] = 100000;
  }

  cudaMemcpy(hist_data, h_hist, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaSetDevice(0);
  cudaDeviceProp deviceProp{};
  cudaGetDeviceProperties(&deviceProp, 0);
  const int blockSize = 256;
  int GridSize = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);
  dim3 Grid(GridSize);
  dim3 Block(blockSize);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  // bug1: L68的N不能传错，之前传的256，导致L19的cache[1]打印出来为0
  histogram<blockSize><<<Grid, Block>>>(hist_data, bin_data, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(bin, bin_data, 256 * sizeof(int), cudaMemcpyDeviceToHost);
  // bug2: 同bug1，L67传进去的256表示两个buffer的数据量，这个必须得精确，之前传的N，尽管只打印第1个值，但依然导致L27打印出来的值为垃圾值
  bool is_right = CheckResult(bin, groudtruth, 256);
  if(is_right) {
	printf("the ans is right\n");
  } else {
	printf("the ans is wrong\n");
	for(int i = 0; i < 256; i++){
	  printf("%d ", bin[i]);
	}
	printf("\n");
  }
  printf("histogram + shared_mem + multi_value latency = %f ms\n", milliseconds);

  cudaFree(bin_data);
  cudaFree(hist_data);
  free(bin);
  free(h_hist);
  free(groudtruth);
}