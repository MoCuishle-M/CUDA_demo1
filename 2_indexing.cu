#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void sum(float *x)
{
  // 泛指当前block在所有block范围内的id
  int block_id = blockIdx.x;
  // 泛指当前线程在所有block范围内的全局id
  int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  // 泛指当前线程在其block内的id
  int local_tid = threadIdx.x;
  printf("current block=%d, thread id in current block =%d, global thread id=%d\n", block_id, local_tid, global_tid);
  x[global_tid] += 1;
}

int main(){
  int N = 32;
  auto size = N * sizeof(float);
  float *d_x, *h_x;
  /* allocate GPU mem */
  /*思考为什么要用二级指针
   * 这里的参数void **devPtr是一个二级指针（或者称为指针的指针），
   * 原因在于cudaMalloc需要修改传入的指针变量，使其指向新分配的设备内存地址。
   * 由于C/C++中函数参数传递是值传递，直接使用一级指针作为参数只能在函数内部修改指针所指向的内容，而不能改变指针本身的值（即它所存储的地址）。
   * 因此，为了能够在调用cudaMalloc后，让主机端的指针变量（如d_x）持有设备内存的地址，就需要使用二级指针。
   * 简而言之，使用二级指针是为了能够让函数外部的指针变量正确地记录下分配的GPU内存地址。
   *
   */
  cudaMalloc((void **)&d_x, size);
  /* allocate CPU mem */
  h_x = (float*) malloc(size);
  /* init host data */
  printf("h_x original: \n");
  for (int i = 0; i < N; i++) {
	h_x[i] = i;
	printf("%g\n", h_x[i]);
  }
  /* copy data to GPU */
  cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
  /* launch GPU kernel */
  sum<<<1, N>>>(d_x);
  /* copy data from GPU */
  cudaMemcpy(h_x, d_x, size, cudaMemcpyDeviceToHost);
  printf("h_x current: \n");
  for (int i = 0; i < N; i++) {
	printf("%g\n", h_x[i]);
  }
  cudaFree(d_x);
  free(h_x);
  return 0;
}