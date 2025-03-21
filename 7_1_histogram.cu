#include <cstdio>
#include <cuda.h>
#include "cuda_runtime.h"
#include <algorithm>
#include <random>

constexpr int kBlockSize = 256;
constexpr int kNumWaves = 32;
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

template <int blockSize>
__global__ void histogram(const int *hist_data, int *bin_data, int N)
{
    __shared__ int cache[blockSize];
    int gtid = blockIdx.x * blockSize + threadIdx.x; // 泛指当前线程在所有block范围内的全局id
    int tid = threadIdx.x; // 泛指当前线程在其block内的id
    cache[tid] = 0; // 每个thread初始化shared mem
    __syncthreads();
    // for循环来自动确定每个线程处理的元素个数
    for (int i = gtid; i < N; i += gridDim.x * blockSize)
    {
        int val = hist_data[i];// 每个单线程计算全局内存中的若干个值
        atomicAdd(&cache[val], 1); // 原子加法，强行使得并行的CUDA线程串行执行加法，但是并不能保证顺序
    }
    __syncthreads();//此刻每个block的bin都已统计在cache这个smem中
    //debug info: if(tid== 0){printf("cache[1]=%d,hist[1]=%d\n",cache[1],hist_data[2]);}
    atomicAdd(&bin_data[tid], cache[tid]);
    //debug info: if(tid== 0){printf("bin_data[1]=%d,hist[1]=%d\n",bin_data[1],hist_data[2]);}
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
    const int blockSize = 256;
    // int GridSize = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);
  	const int64_t GridSize = GetNumBlocks(N);
	printf("GridSize: %lld\n", GridSize);
    dim3 Grid(GridSize);
    dim3 Block(blockSize);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

  	histogram<blockSize><<<Grid, Block>>>(d_hist_data, d_bin_data, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_bin, d_bin_data, 256 * sizeof(int), cudaMemcpyDeviceToHost);
    // bug2: 同bug1，L67传进去的256表示两个buffer的数据量，这个必须得精确，之前传的N，尽管只打印第1个值，但依然导致L27打印出来的值为垃圾值
    bool is_right = CheckResult(h_bin, groudtruth, 256);
    if(is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        for(int i = 0; i < 256; i++){
            printf("%d ", h_bin[i]);
        }
        printf("\n");
    }
    printf("histogram + shared_mem + multi_value latency = %f ms\n", milliseconds);    

    cudaFree(d_bin_data);
    cudaFree(d_hist_data);
    free(h_bin);
    free(h_hist);
   	free(groudtruth);
}
