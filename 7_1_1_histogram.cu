#include <cstdio>
#include <cuda.h>
#include "cuda_runtime.h"
#include <algorithm>
#include <random>
#include <math_functions.h>
constexpr int kBlockSize = 256;
constexpr int kNumWaves = 32;
/*这里的意思就是我们还可以通过调整GridSize和BlockSize的方式获得更好的性能收益，
 *也就是说一个线程负责更多的元素计算
 */
int64_t GetNumBlocks(int64_t n) {
  int dev;
  {
	cudaError_t err = cudaGetDevice(&dev);
	if (err != cudaSuccess) { return err; }
  }
  // SM的数量
  int sm_count;
  {
	cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
	printf("sm_count: %d\n", sm_count);
	if (err != cudaSuccess) { return err; }
  }
  // 每个SM的线程最大数量
  int tpm;
  {
	cudaError_t err = cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
	printf("Max Threads Per Multi Processor: %d\n", tpm);
	if (err != cudaSuccess) { return err; }
  }
  //
  int64_t num_blocks = std::max<int64_t>(1, std::min<int64_t>((n + kBlockSize - 1) / kBlockSize,
															  sm_count * tpm / kBlockSize * kNumWaves));
  return num_blocks;
}

template <unsigned int blockSizeX, unsigned int blockSizeY>
__global__ void histogram(const int *hist_data, int *bin_data, int N)
{
  __shared__ int cache[blockSizeX][blockSizeY+1];
  cache[threadIdx.x][threadIdx.y] = 0;
  // 全局索引
  unsigned int idx = blockIdx.x * blockSizeX * blockSizeY + threadIdx.y * blockSizeX + threadIdx.x;
  for (auto i = idx; i < N; i += gridDim.x * blockSizeX * blockSizeY)
  {
	int val = hist_data[i];
	// X/2的幂次方  ==  X>>log2(blockSizeX)
	int rowIndex = val >> 4;
	int colIndex = val & (blockSizeY-1);
	atomicAdd(&cache[rowIndex][colIndex], 1);
  }
  __syncthreads();
  atomicAdd(&bin_data[threadIdx.y * blockSizeX + threadIdx.x], cache[threadIdx.x][threadIdx.y]);
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
  int *h_bin = (int *)malloc(256 * sizeof(int));
  int *d_bin_data;
  int *d_hist_data;
  cudaMalloc((void **)&d_bin_data, 256 * sizeof(int));
  cudaMalloc((void **)&d_hist_data, N * sizeof(int));

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

  cudaMemcpy(d_hist_data, h_hist, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaSetDevice(0);
  cudaDeviceProp deviceProp{};
  cudaGetDeviceProperties(&deviceProp, 0);
  const int64_t GridSize = GetNumBlocks(N);
  //const int64_t GridSize = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);
  printf("GridSize: %lld\n", GridSize);
  dim3 Grid(GridSize);
  dim3 Block(16,16);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  histogram<16, 16><<<Grid, Block>>>(d_hist_data, d_bin_data, N);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(h_bin, d_bin_data, 256 * sizeof(int), cudaMemcpyDeviceToHost);
  for(int i = 0; i < 256; i++){
	printf("%d \n", h_bin[i]);
  }

  bool is_right = CheckResult(h_bin, groudtruth, 256);
  if(is_right) {
	printf("the ans is right\n");
  } else {
	printf("the ans is wrong\n");
//	for(int i = 0; i < 256; i++){
//	  printf("%d ", h_bin[i]);
//	}
	printf("\n");
  }
  printf("histogram + shared_mem + multi_value latency = %f ms\n", milliseconds);

  cudaFree(d_bin_data);
  cudaFree(d_hist_data);
  free(h_bin);
  free(h_hist);
  free(groudtruth);
}
