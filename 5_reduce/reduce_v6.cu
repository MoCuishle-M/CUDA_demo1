#include <cstdio>
#include <cuda.h>
#include "cuda_runtime.h"

constexpr int kBlockSize = 256;
constexpr int kNumWaves = 1;
/*这里的意思就是我们还可以通过调整GridSize和BlockSize的方式获得更好的性能收益，也就是说一个线程负责更多的元素计算*/
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

// 注意: v0-v5里面kernel得到的是各个block负责范围内的总和，要想得到最终的和，需要把各个block求得的总和再做reduce sum
// v6: multi-block reduce final result by two pass
template <int blockSize>
__device__ void BlockSharedMemReduce(float* smem) {
    //对v4 L45的for循环展开，以减去for循环中的加法指令，以及给编译器更多重排指令的空间
  if (blockSize >= 1024) {
    if (threadIdx.x < 512) {
      smem[threadIdx.x] += smem[threadIdx.x + 512];
    }
    __syncthreads();
  }
  if (blockSize >= 512) {
    if (threadIdx.x < 256) {
      smem[threadIdx.x] += smem[threadIdx.x + 256];
    }
    __syncthreads();
  }
  if (blockSize >= 256) {
    if (threadIdx.x < 128) {
      smem[threadIdx.x] += smem[threadIdx.x + 128];
    }
    __syncthreads();
  }
  if (blockSize >= 128) {
    if (threadIdx.x < 64) {
      smem[threadIdx.x] += smem[threadIdx.x + 64];
    }
    __syncthreads();
  }
  // the final warp
  if (threadIdx.x < 32) {
	volatile float* vshm = smem;
	float x = vshm[threadIdx.x];
	if (blockDim.x >= 64) {
	  x += vshm[threadIdx.x+32];__syncwarp();
	  vshm[threadIdx.x] = x;__syncwarp();
	}
	x += vshm[threadIdx.x+16];__syncwarp();
	vshm[threadIdx.x] = x;__syncwarp();
	x += vshm[threadIdx.x+8];__syncwarp();
	vshm[threadIdx.x] = x;__syncwarp();
	x += vshm[threadIdx.x+4];__syncwarp();
	vshm[threadIdx.x] = x;__syncwarp();
	x += vshm[threadIdx.x+2];__syncwarp();
	vshm[threadIdx.x] = x;__syncwarp();
	x += vshm[threadIdx.x+1];__syncwarp();
	vshm[threadIdx.x] = x;__syncwarp();
  }
}

template <int blockSize>
__global__ void reduce_v6(const float *d_in, float *d_out, int64_t nums){
    __shared__ float smem[blockSize];
    // 泛指当前线程在其block内的id
    unsigned int tid = threadIdx.x;
    // 泛指当前线程在所有block范围内的全局id
    unsigned int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_thread_num = blockDim.x * gridDim.x;
    // 基于v5的改进：不用显式指定一个线程处理2个元素，而是通过L94的for循环来自动确定每个线程处理的元素个数
    float sum = 0.0f;
    for (auto i = gtid; i < nums; i += total_thread_num) {
        sum += d_in[i];
    }
    smem[tid] = sum;
    __syncthreads();
    // compute: reduce in shared mem
    BlockSharedMemReduce<blockSize>(smem);

    // store: 哪里来回哪里去，把reduce结果写回显存
    // GridSize个block内部的reduce sum已得出，保存到d_out的每个索引位置
    if (tid == 0) {
        d_out[blockIdx.x] = smem[0];
    }
}

bool CheckResult(const float *out, float groundtruth, int n){
    if (*out != groundtruth) {
      return false;
    }
    return true;
}

int main(){
    cudaDeviceProp deviceProp{};
    cudaGetDeviceProperties(&deviceProp, 0);
    int maxblocks = deviceProp.maxGridSize[0];
	printf("Device name: %s\n", deviceProp.name);
	printf("Maximum size of each dimension of a grid: %d\n", maxblocks);
    const int blockSize = 256;
    const int64_t N = 25600000;
    //int gridSize = std::min((N + blockSize - 1) / blockSize, maxblocks);

	// 通过v7中int64_t GetNumBlocks(int64_t n)函数得到的gridSize
  	const int64_t gridSize = GetNumBlocks(N);

    float milliseconds = 0;
    auto *a = (float *)malloc(N * sizeof(float));
    float *d_a;
    cudaMalloc((void **)&d_a,N * sizeof(float));

    auto *out = (float*)malloc((gridSize) * sizeof(float));
    float *d_out;
    float *part_out;//新增part_out存储每个block reduce的结果
    cudaMalloc((void **)&d_out, 1 * sizeof(float));
    cudaMalloc((void **)&part_out, (gridSize) * sizeof(float));
    auto ground_truth = static_cast<float>(N);

    for(int i = 0; i < N; i++){
        a[i] = 1;
    }

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(gridSize);
    dim3 Block(blockSize);
    

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    reduce_v6<blockSize><<<Grid, Block>>>(d_a, part_out, N);
    reduce_v6<blockSize><<<1, Block>>>(part_out, d_out, gridSize);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(out, d_out, 1 * sizeof(float), cudaMemcpyDeviceToHost);
    bool is_right = CheckResult(out, ground_truth, 1);
    if(is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        for(int i = 0;i < 1;i++){
            printf("%lf ",out[i]);
        }
        printf("\n");
    }
    printf("reduce_v6 latency = %f ms\n", milliseconds);
  	printf("out[0]: %lf\n",out[0]);
  	printf("res[0]: %lf\n", ground_truth);
    cudaFree(d_a);
    cudaFree(d_out);
    cudaFree(part_out);
    free(a);
    free(out);
}
