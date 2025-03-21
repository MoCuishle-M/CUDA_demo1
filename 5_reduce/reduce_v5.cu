﻿#include <cstdio>
#include <cuda.h>
#include "cuda_runtime.h"

#define THREAD_PER_BLOCK 256

// v5：循环展开
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
__global__ void reduce_v5(const float *d_in, float *d_out){
    __shared__ float smem[blockSize];
    // 泛指当前线程在其block内的id
    unsigned int tid = threadIdx.x;
    // 泛指当前线程在所有block范围内的全局id, *2代表当前block要处理2*blocksize的数据
  	// ep. blocksize = 2, blockIdx.x = 0, when threadIdx.x = 0, gtid = 0, gtid + blockSize = 2;
	//                                    when threadIdx.x = 1, gtid = 1, gtid + blockSize = 3
    // ep. blocksize = 2, blockIdx.x = 1, when threadIdx.x = 0, gtid = 4, gtid + blockSize = 6;
	//                                    when threadIdx.x = 1, gtid = 5, gtid + blockSize = 7
    unsigned int i = blockIdx.x * (blockSize * 2) + threadIdx.x;
    // load: 每个线程加载两个元素到shared mem对应位置
    smem[tid] = d_in[i] + d_in[i + blockSize];
    __syncthreads();
    // compute: reduce in shared mem
    BlockSharedMemReduce<blockSize>(smem);

    // store: 哪里来回哪里去，把reduce结果写回显存
    // GridSize个block内部的reduce sum已得出，保存到d_out的每个索引位置
    if (tid == 0) {
        d_out[blockIdx.x] = smem[0];
    }
}

bool CheckResult(const float *out, float groudtruth, int n){
    float res = 0;
    for (int i = 0; i < n; i++){
        res += out[i];
    }
    if (res != groudtruth) {
        return false;
    }
    return true;
}

int main(){
    float milliseconds = 0;
    
    const int N = 25600000;
    cudaSetDevice(0);
    cudaDeviceProp deviceProp{};
    cudaGetDeviceProperties(&deviceProp, 0);
    int GridSize = std::min((N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK/2, deviceProp.maxGridSize[0]);
    //int GridSize = 100000;
    auto *a = (float *)malloc(N * sizeof(float));
    float *d_a;
    cudaMalloc((void **)&d_a, N * sizeof(float));

    auto *out = (float*)malloc((GridSize) * sizeof(float));
    float *d_out;
    cudaMalloc((void **)&d_out, (GridSize) * sizeof(float));

    for(int i = 0; i < N; i++){
        a[i] = 1.0f;
    }

    float groudtruth = N * 1.0f;

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(GridSize);
    dim3 Block(THREAD_PER_BLOCK);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    reduce_v5<THREAD_PER_BLOCK><<<Grid,Block>>>(d_a, d_out);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(out, d_out, GridSize * sizeof(float), cudaMemcpyDeviceToHost);
    printf("allcated %d blocks, data counts are %d \n", GridSize, N);
    bool is_right = CheckResult(out, groudtruth, GridSize);
    if(is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        for(int i = 0; i < GridSize;i++){
            printf("resPerBlock : %lf ",out[i]);
        }
        printf("\n");
        printf("groudtruth is: %f \n", groudtruth);
    }
    printf("reduce_v5 latency = %f ms\n", milliseconds);

    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
}
