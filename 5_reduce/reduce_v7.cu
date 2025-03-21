#include <cuda.h>
#include <cuda_runtime.h>
#include <ctime>
#include <algorithm>
#include <cuda_fp16.h>
#include <cstdio>

#define PackSize 4
#define kWarpSize 32
#define N 25600000
constexpr int BLOCK_SIZE = 256;

constexpr int kBlockSize = 256;
constexpr int kNumWaves = 32;
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


/*
 * pack data的模板struct，pack_size 是打包在一起的元素个数
 * alignas(sizeof(T) * pack_size) :确保结构体在内存中按照 T 类型的 pack_size 元素的大小对齐。
 * 成员函数存储在类的类型信息部分，这部分通常称为“类的静态存储区域”或者简单地说是程序的代码段，
 * 当一个对象调用这个成员函数时，实际上是通过该对象的指针（或引用）间接访问这个共享的函数代码。对象的内存布局只包含非静态数据成员，不包括函数。
 */
template<typename T, int pack_size>
struct alignas(sizeof(T) * pack_size) Packed {
  __device__ explicit Packed(T val){
#pragma unroll
	for(int i = 0; i < pack_size; i++){
	  elem[i] = val;
	}
  }
  __device__ Packed() {
	// do nothing
  }
  // 不明白为什么这里要用union，方便扩展？
  union {
	T elem[pack_size];
  };
  __device__ void operator+=(Packed<T, pack_size> packA){
#pragma unroll
	for(int i = 0; i < pack_size; i++){
	  elem[i] += packA.elem[i];
	}
  }
};

template<typename T, int pack_size>
__device__ T PackReduce(Packed<T, pack_size> pack){
  T res = 0.0;
#pragma unroll
  for(int i = 0; i < pack_size; i++){
	res += pack.elem[i];
  }
  return res;
}

template<typename T>
__device__ T warpReduceSum(T val){
  for(int lane_mask = 16; lane_mask > 0; lane_mask >>=1){
	val += __shfl_down_sync(0xffffffff, val, lane_mask);
  }
  return val;
}

__global__ void reduce_v7(const float *g_idata, float *g_odata, unsigned int n){

  // each thread loads one element from global to shared mem

  unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;
  // [0.0, 0.0, 0.0, 0.0]
  Packed<float, PackSize> sum_pack(0.0);
  Packed<float, PackSize> load_pack(0.0);
  // 改变指针类型，也即g_idata指向的内存区域将按照Packed<float, PackSize>的布局重新解释。
  const auto* pack_ptr = reinterpret_cast<const Packed<float, PackSize>*>(g_idata);

  // 每个线程负责从全局内存加载PackSize个元素，并累加到sum_pack中
  // 这里需要使用for循环，因为每个线程负责的元素个数是不确定的，需要根据n来确定
  for(int32_t linear_index = gid; linear_index < n / PackSize; linear_index+=blockDim.x * gridDim.x){
	Packed<float, PackSize> g_idata_load = pack_ptr[linear_index];
	sum_pack += g_idata_load;
  }
  auto PackReduceVal = PackReduce<float, PackSize>(sum_pack);
  // Shared mem for partial sums (one per warp in the block)
  static __shared__ float warpLevelSums[kWarpSize];
  const int laneId = threadIdx.x % kWarpSize;
  const int warpId = threadIdx.x / kWarpSize;

  auto sum = warpReduceSum<float>(PackReduceVal);
  __syncthreads();

  if(laneId == 0 )warpLevelSums[warpId] = sum;
  __syncthreads();
  // read from shared memory only if that warp existed
  sum = (threadIdx.x < blockDim.x / kWarpSize) ? warpLevelSums[laneId] : 0;
  // Final reduce using first warp
  if (warpId == 0) sum = warpReduceSum<float>(sum);
  // write result for this block to global mem
  if (threadIdx.x == 0) g_odata[blockIdx.x] = sum;
}

bool check(const float *out,const float *res,int n){
  for(int i=0;i<n;i++){
	if(out[i]!=res[i])
	  return false;
  }
  return true;
}

int main(){
  auto *a=(float *)malloc(N*sizeof(float));
  float *d_in_data;
  cudaMalloc((void **)&d_in_data, N*sizeof(float));

  const int64_t gridSize = GetNumBlocks(N);
  printf("Block num is: %lld \n", gridSize);

  auto *out=(float *)malloc(sizeof(float));
  float *g_odata;
  cudaMalloc((void **)&g_odata, gridSize*sizeof(float));
  float *g_final_data;
  cudaMalloc((void **)&g_final_data,1*sizeof(float));

  for(int i=0;i<N;i++){
	a[i]=1;
  }
  auto *res=(float *)malloc(sizeof(float));
  res[0] = N * (a[0] + a[N - 1]) / 2;

  cudaMemcpy(d_in_data, a, N*sizeof(float), cudaMemcpyHostToDevice);;
  float milliseconds = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  reduce_v7<<<gridSize, BLOCK_SIZE>>>(d_in_data, g_odata, N);
  reduce_v7<<<1, BLOCK_SIZE>>>(g_odata, g_final_data, gridSize);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(out,g_final_data,1*sizeof(float),cudaMemcpyDeviceToHost);

  if(check(out,res,1))printf("the answer is right\n");
  else{
	printf("the answer is wrong\n");
	printf("out[0]: %lf\n",out[0]);
	printf("res[0]: %lf\n", res[0]);
  }
  printf("out[0]: %lf\n",out[0]);
  printf("res[0]: %lf\n", res[0]);
  printf("reduce_v7 latency = %f ms\n", milliseconds);

  cudaFree(d_in_data);
  cudaFree(g_odata);
  free(a);
  free(out);
  free(res);
  return 0;
}