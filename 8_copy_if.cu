#include <cstdio>
#include <cuda.h>
#include "cuda_runtime.h"
#include "cooperative_groups.h"
//#define THREAD_PER_BLOCK 256
// 这种warp和shared在老的gpu上面会很有成效，但是在turing后的GPU，nvcc编译器优化了很多，所以导致效果不明显
// cpu
int filter(int *dst, int *src, int n) {
  int nres = 0;
  for (int i = 0; i < n; i++)
    if (src[i] > 0)
      dst[nres++] = src[i];
  // return the number of elements copied
  return nres;
}
// 数据量为256000000时，latency=14.37ms
// naive kernel
//__global__ void filter_k(int *dst, int *nres, int *src, int n) {
//  int i = threadIdx.x + blockIdx.x * blockDim.x;
//  // 输入数据大于0的，计数器+1，并把该数写到输出显存以计数器值为索引的地址
//  if(i < n && src[i] > 0)
//    dst[atomicAdd(nres, 1)] = src[i];
//}

// 数据量为256000000时，latency=13.86ms
// block level, use block level atomics based on shared memory
 __global__ void filter_shared_k(int *dst, int *nres, const int* src, int n) {
   // 计数器声明为shared memory，去计数各个block范围内大于0的数量
   __shared__ int l_n;
   int gtid = blockIdx.x * blockDim.x + threadIdx.x;
   int total_thread_num = blockDim.x * gridDim.x;

   for (int i = gtid; i < n; i += total_thread_num) {
     // use first thread to zero the counter
     // 初始化只需1个线程来操作
     if (threadIdx.x == 0)
       l_n = 0;
     __syncthreads();

     int d, pos;
     // l_n表示每个block范围内大于0的数量，block内的线程都可访问
     // pos是每个线程私有的寄存器，且作为atomicAdd的返回值，表示当前线程对l_n原子加1之前的l_n，比如1 2 4号线程都大于0，那么对于4号线程来说l_n = 3, pos = 2
     if(i < n) {
	   d = src[i];
	   if(d > 0){
		 pos = atomicAdd(&l_n, 1);
	   }
     }
     __syncthreads();

     // 每个block选出tid=0作为leader
     // leader把每个block的l_n累加到全局计数器(nres),即所有block的局部计数器做一个reduce sum
     // 注意: 下下行原子加返回的l_n为全局计数器nres原子加l_n之前的nres，比如对于block1，已知原子加前，nres = 2, l_n = 3，原子加后, nres = 2+3, 返回的l_n = 2
     if(threadIdx.x == 0)
       l_n = atomicAdd(nres, l_n);
     __syncthreads();

     //write & store
     if(i < n && d > 0) {
     // 1. pos: src[thread]>0的thread在当前block的index
     // 2. l_n: 在当前block的前面几个block的所有src>0的个数
     // 3. pos + l_n：当前thread的全局offset
       pos += l_n;
       dst[pos] = d;
     }
     __syncthreads();
   }
 }


//warp level, use warp-aggregated atomics
__device__ int atomicAggInc(int *ctr) {
  // 获取当前wa掩rp中所有活跃线程的码。
  unsigned int active = __activemask();
  // leader应该表示warp里面第一个src[threadIdx.x]>0的threadIdx.x
  // __ffs()找出 x 中设置为 1 的第一位（最小有效位）的位置，其中最小有效位的位置是 1。
  // 例如，对于输入 00010010，__ffs() 返回 2。
  int leader = __ffs(active) - 1;
  //warp mask中为1的数量
  int change = __popc(active);
  int lane_mask_lt;
  // 获取小于当前线程ID的掩码。
  /*
    asm：内联汇编指令。
	"mov.u32 %0, %%lanemask_lt;"：将 %%lanemask_lt 的值移动到目标寄存器 %0。%%lanemask_lt 是一个特殊寄存器，表示当前线程 lane 之前所有线程的掩码。
	: "=r"(lane_mask_lt)：输出操作数约束。=r 表示输出操作数是一个通用寄存器，lane_mask_lt 是目标变量。
   e.g.
  		当前线程是第 3 个线程（threadIdx.x % warpSize == 3），那么 %%lanemask_lt 的值为 0b00000111（即前三个线程的掩码）。
  */
  asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lane_mask_lt));
  // 比当前线程id小且值为1的mask之和 -> 计算当前线程在warp中的位置
  unsigned int rank = __popc(active & lane_mask_lt);
  int warp_res;
  if(rank == 0)//leader thread of every warp
	// atomicAdd 返回在执行增加操作之前，内存地址 address 处的原始数值。
	// 这里相当于得到每个warp要开始输出值的位置，可以理解为数组指针
    warp_res = atomicAdd(ctr, change);//compute global offset of warp
  warp_res = __shfl_sync(active, warp_res, leader);//broadcast warp_res of leader thread to every active thread
  return warp_res + rank; // global offset + local offset = final offset，即L91表示的atomicAggInc(nres), 为src[i]的最终的写入到dst的位置
}

__global__ void filter_warp_k(int *dst, int *nres, const int *src, int n) {
  // 计算当前线程在全局中的索引。
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if(i >= n)
    return;
  if(src[i] > 0) // 过滤出src[i] > 0 的线程，比如warp0里面只有0号和1号线程的src[i]>0，那么只有0号和1号线程运行L91，对应的L72的__activemask()为110000...00
    // 通过atomicAggInc函数计算当前线程负责数据的全局偏移量，并将满足条件的元素写入目标数组。
    dst[atomicAggInc(nres)] = src[i];
}

bool CheckResult(const int *out, int groudtruth, int n){
    if (*out != groudtruth) {
        return false;
    }
    return true;
}

int main(){
    float milliseconds = 0;
    int N = 2560000;

    cudaSetDevice(0);
    cudaDeviceProp deviceProp{};
    cudaGetDeviceProperties(&deviceProp, 0);
    const int blockSize = 256;
    int GridSize = std::min((N + 256 - 1) / 256, deviceProp.maxGridSize[0]);

    int *src_h = (int *)malloc(N * sizeof(int));
    int *dst_h = (int *)malloc(N * sizeof(int));
    int *nres_h = (int *)malloc(1 * sizeof(int));
    int *dst, *nres;
    int *src;
    cudaMalloc((void **)&src, N * sizeof(int));
    cudaMalloc((void **)&dst, N * sizeof(int));
    cudaMalloc((void **)&nres, 1 * sizeof(int));

    for(int i = 0; i < N; i++){
        src_h[i] = 1;
    }

    int groudtruth = 0;
    for(int j = 0; j < N; j++){
        if (src_h[j] > 0) {
            groudtruth += 1;
        }
    }


    cudaMemcpy(src, src_h, N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 Grid(GridSize);
    dim3 Block(blockSize);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    //filter_warp_k<<<Grid, Block>>>(dst, nres, src, N);
    filter_shared_k<<<Grid, Block>>>(dst, nres, src, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(nres_h, nres, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    bool is_right = CheckResult(nres_h, groudtruth, N);
    if(is_right) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        printf("%lf ",*nres_h);
        printf("\n");
    }
    printf("filter_k latency = %f ms\n", milliseconds);    

    cudaFree(src);
    cudaFree(dst);
    cudaFree(nres);
    free(src_h);
    free(dst_h);
    free(nres_h);
}
