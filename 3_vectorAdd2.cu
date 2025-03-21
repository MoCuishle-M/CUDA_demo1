#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

typedef float FLOAT;

/* CUDA kernel function */
__global__ void vec_add(FLOAT *x, FLOAT *y, FLOAT *z, int N)
{
  /* 2D grid */
  int idx = (blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x) + threadIdx.x);
  /* 1D grid */
  // int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < N) z[idx] = y[idx] + x[idx];
}

void vec_add_cpu(FLOAT *x, FLOAT *y, FLOAT *z, int N)
{
  for (int i = 0; i < N; i++) z[i] = y[i] + x[i];
}

int main()
{
  int N = 10000;
  int size = N * sizeof(FLOAT);

  /* 1D block */
  int block_size = 256;

  /* 2D grid */
  // 向上取整，再开方，向上取整
  int s = ceil(sqrt((N + block_size - 1.) / block_size));
  printf("s = %d\n", s);
  dim3 grid(s, s);
  /* 1D grid */
  // int s = ceil((N + block_size - 1.) / block_size);
  // dim3 grid(s);

  FLOAT *d_x, *h_x;
  FLOAT *d_y, *h_y;
  FLOAT *d_z, *h_z;

  /* allocate GPU mem */
  cudaMalloc((void **)&d_x, size);
  cudaMalloc((void **)&d_y, size);
  cudaMalloc((void **)&d_z, size);

  /* init time */
  float milliseconds = 0;

  /* alllocate CPU mem */
  h_x = (FLOAT *) malloc(size);
  h_y = (FLOAT *) malloc(size);
  h_z = (FLOAT *) malloc(size);

  /* init */
  for (int i = 0; i < N; i++) {
	h_x[i] = 1;
	h_y[i] = 1;
  }

  /* copy data to GPU */
  cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  /* launch GPU kernel */
  vec_add<<<grid, block_size>>>(d_x, d_y, d_z, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);

  /* copy GPU result to CPU */
  cudaMemcpy(h_z, d_z, size, cudaMemcpyDeviceToHost);

  /* CPU compute */
  FLOAT* hz_cpu_res = (FLOAT *) malloc(size);
  vec_add_cpu(h_x, h_y, hz_cpu_res, N);

  /* check GPU result with CPU*/
  for (int i = 0; i < N; ++i) {
	if (fabs(hz_cpu_res[i] - h_z[i]) > 1e-6) {
	  printf("Result verification failed at element index %d!\n", i);
	}
  }
  printf("Result right\n");
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);

  free(h_x);
  free(h_y);
  free(h_z);
  free(hz_cpu_res);

  return 0;
}