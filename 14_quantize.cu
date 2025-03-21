#include <cmath>
#include <cfenv>
#include <random>
#include <cstdio>
#include <cfloat>
#include <cuda.h>
#include "cuda_runtime.h"
// Kernel performance:
// PerTensor + Sym: 0.58ms
// PerChannel + Sym: 0.46ms
// solved bugs:
// 1. gpu_res[0] = 0, cpu_res[0] = 86
//     cpu_max is right, gpu_max = very big
// 2. cpu_min != gpu_min, cpu_max != gpu_max, check minmax kernel and guess it resulted from cuda kernel某些地方写错 or atomicMax
// 3. cpu_scale != gpu_scale, ep. cpu_scale = 0.22035, gpu_scale = 0.22036
// 4. cpu_res != gpu_res , ep. cpu_res = 44, gpu_res = 45
// debug tips:
// 1. input some simple case to confirm cpu impl is right
// 1. printf cpu res and gpu res of each kernel
// 2. use if(tid==0) to get the gpu output and key variable of one thread
// 3. use grid step loop to conveniently debug by launch one thread

bool CheckResult(float *out, float* groudtruth, int nums){
    for (int i = 0; i < nums; i++){
      if (groudtruth[i] != out[i]) {
        printf("the wrong index is %d, the groudtruth is %f, the res is %f\n", i, groudtruth[i], out[i]);
        return false;
      }
    }
    return true;
}
// CPU version equation
// PerTensor + Sym: scale = max(abs(weight)) / 127 , zeropoint = 0, input_int8 = clamp(input_fp32/scale ,-128, 127)
// PerTensor + Asym: scale = (max(weight) - min(weight)) / 255, zeropoint = -round(min(weight))/scale - 2的b-1次方
// PerChannel + Sym: scale[channel_id] = max(abs(weight[channel_id])) / 127 , zeropoint = 0, input_int8[channel_id * HW + (channel_id + 1) * HW] = clamp(input_fp32[channel_id * HW + (channel_id + 1) * HW]/scale[channel_id] ,-128, 127)
// PerChannel + Asym: scale[channel_id] = (max(weight[channel_id]) - min(weight[channel_id])) / 255, zeropoint[channel_id] = -round(min(weight[channel_id]))/scale[channel_id] - 2的b-1次方

// py code
// def gen_quant_scale_for_min_max_symmetric(weight, quantization_bit):
//     weight_max = np.max(np.abs(weight))
//     denominator = 2.0 ** (quantization_bit - 1) - 1
//     return (weight_max / denominator, 0)

/**
 * @brief 生成针对每个张量的对称量化参数
 *
 * 该函数计算给定输入数据的量化解参数，包括量化解的缩放因子和零点。
 * 它适用于对每个张量进行对称量化的情况，其中零点固定为0，缩放因子根据输入数据的范围计算。
 *
 * @tparam T 输入数据和量化参数的类型
 * @param in_ptr 输入数据的指针
 * @param quantization_bit 量化位数
 * @param num_elements 输入数据中的元素数量
 * @param scale 输出缩放因子的指针
 * @param zero_point 输出零点的指针
 */
template<typename T>
void GenScalePerTensorSymmetricCPU(const T* in_ptr, const int quantization_bit,
                                   const int num_elements, T* scale, T* zero_point) {
  // 计算输入数据中的最大值和最小值
  T in_max = *std::max_element(in_ptr, in_ptr + num_elements);
  T in_min = *std::min_element(in_ptr, in_ptr + num_elements);

  // 确定量化后的输出数据的最大绝对值
  T out_max = std::max(std::abs(in_max), std::abs(in_min));

  // 根据量化位数计算量化的分母
  T denominator = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;

  // 计算缩放因子
  *scale = out_max / denominator;

  // 零点固定为0
  *zero_point = 0;
}
// py code
// def gen_quant_scale_for_min_max_affine(weight, quantization_bit):
//     weight_max = np.max(weight)
//     weight_min = np.min(weight)
//     denominator = 2.0 ** quantization_bit - 1
//     scale = (weight_max - weight_min) / denominator
//     zero_point = -np.round(weight_min / scale)
//     return (scale, zero_point)

// 公式: clip(input / scale .round(), -128, 127)
/**
 * @brief 对每个张量进行对称量化
 *
 * 该函数根据给定的缩放因子和量化位数，将输入数据量化为指定位数的量化值。
 * 它适用于对每个张量进行对称量化的情况，其中零点固定为0。
 *
 * @tparam T 输入数据和量化结果的类型
 * @param in_ptr 输入数据的指针
 * @param scale 缩放因子
 * @param quantization_bit 量化位数
 * @param num_elements 输入数据中的元素数量
 * @param out_ptr 量化结果的输出指针
 */
template<typename T>
void QuantizationPerTensorSymmetricCPU(const T* in_ptr, const T scale, const int quantization_bit,
                                       const int num_elements, T* out_ptr) {
  // 计算量化的上限和下限
  T upper_bound = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;
  T lower_bound = -upper_bound - 1;

  // 遍历每个元素进行量化
  for(int j = 0; j < num_elements; j++) {
    // 计算量化值，使用nearbyint函数确保结果是附近的整数
    T out = std::nearbyint(in_ptr[j] / scale);

    // 限制量化值在合法的范围内
    out = out > upper_bound ? upper_bound : out;
    out = out < lower_bound ? lower_bound : out;

    // 将量化值写入输出数组
    out_ptr[j] = out;
  }
}

// def quant_per_layer_affine(input, quantization_bit, scale, zero_point):
//     upper_bound = 2.0 ** quantization_bit - 1
//     lower_bound = 0
//     return np.clip(np.rint(input / scale + zero_point), lower_bound, upper_bound)

/**
 * @brief 计算每个通道的对称量化尺度和零点
 *
 * 该函数针对给定的输入数据，在每个通道上计算量化时使用的尺度（scale）和零点（zero_point）。
 * 它假设量化是基于每个通道的，并且采用对称量化方案。对称量化意味着正负值将被映射到相同的量化级别。
 *
 * @tparam T 输入数据和输出尺度、零点的数据类型
 * @param in_ptr 输入数据的指针
 * @param quantization_bit 量化位数，用于计算尺度
 * @param HW 输入数据的高度和宽度的乘积，表示每个通道的元素数量
 * @param channel 输入数据的通道数
 * @param num_elements 输入数据的总元素数量
 * @param scale 存储每个通道的量化尺度的输出数组
 * @param zero_point 存储每个通道的量化零点的输出数组
 */
template<typename T>
void GenScalePerChannelSymmetricCPU(const T* in_ptr, const int quantization_bit, const int HW, const int channel,
                                    const int num_elements, T* scale, T* zero_point) {
  // 遍历每个通道，计算该通道的最大和最小值
  // 与per tensor唯一不同在于每个channel的scale不同，所以循环求scale
  for (int cid = 0; cid < channel; cid++){
    // 计算当前通道的起始和结束位置
    int start = cid * HW;
    int end = (cid + 1) * HW;
    // 计算当前通道的最大和最小值
    T channel_max = *std::max_element(in_ptr + start, in_ptr + end); // absmax
    T channel_min = *std::min_element(in_ptr + start, in_ptr + end);// note: cannot use [] which get a float, must use + to get pointer
    // 计算当前通道的量化范围的最大值
    T out_max = std::max(std::abs(channel_max), std::abs(channel_min));
    // 计算量化尺度，使用2的幂次公式进行计算
    T denominator = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;
    scale[cid] = out_max / denominator;
    // 对于对称量化，零点设置为0
    zero_point[cid] = 0;
  }
}
/**
 * @brief 对输入数据进行通道内对称量化处理
 *
 * 该函数针对每个通道内的数据，根据给定的量化位数和缩放因子，进行对称量化处理。
 * 量化的过程是将浮点数映射到一个有限的整数集，通常用于深度学习模型的量化优化。
 *
 * @tparam T 输入和输出数据的类型，通常为float或int
 * @param in_ptr 输入数据的指针
 * @param scale 每个通道的缩放因子指针，用于量化计算
 * @param quantization_bit 量化位数，决定量化后的整数集大小
 * @param HW 每个通道的数量，用于计算通道索引
 * @param num_elements 需要处理的总元素数量
 * @param out_ptr 输出数据的指针
 */
template<typename T>
void QuantizationPerChannelSymmetricCPU(const T* in_ptr, const T* scale, const int quantization_bit, const int HW,
                                        const int num_elements, T* out_ptr) {
  // 计算量化范围的上限和下限
  T upper_bound = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;
  T lower_bound = -upper_bound - 1;

  // 遍历所有元素，进行量化处理
  // j / HW索引到当前元素的channel ID，然后取出对应channel的scale，做quantize
  for(int j = 0; j < num_elements; j++) {
    // 根据当前元素和对应的通道缩放因子进行量化计算
    T out = std::nearbyint(in_ptr[j] / scale[j / HW]);

    // 确保量化结果在有效范围内
    out = out > upper_bound ? upper_bound : out;
    out = out < lower_bound ? lower_bound : out;

    // 将量化后的结果写入输出数组
    out_ptr[j] = out;
  }
}
// 以上是CPU上的quantize函数，接下来用CUDA重写
// GPU device function
__device__ float gpuNearbyint(float a) {
  return std::nearbyint(a);
}

//about CAS: https://blog.csdn.net/m0_52153904/article/details/130095643
//int atomicCAS(int* address, int compare, int val)
//{
//    old = *address;
//    if(old == compare)
//        *address = val;
//    else
//        *address = old;
//    return(old);
//}

// Note: another version of float type atomicMax
//inline __device__ float atomicMax(float *addr, float value) {
//  float old = *addr, assumed;
//  if (old >= value) return old;
//  do {
//    assumed = old;
//    if (assumed > value) break;
//    old = atomicCAS((unsigned int *)addr, __float_as_int(assumed),
  //                  __float_as_int(value));

//  } while (old != assumed);

//  return old;
//}
// 封装好的atmoicMax不支持fp32类型，所以我们这里需要针对fp32类型重载atomicMax
// fp32 type atomicMax from stackoverflow and nv developer forum: https://forums.developer.nvidia.com/t/cuda-atomicmax-for-float/194207
/**
 * 在GPU设备上执行原子最大值操作。
 *
 * 该函数用于比较并更新指定地址上的浮点数，将其更新为输入值和当前值中的最大值。
 * 这是一个原子操作，确保在多线程环境下不会出现竞争条件。
 *
 * @param address 指向要比较和更新的浮点数的地址。
 * @param val 要与当前值比较的最大值。
 * @return 返回更新前的旧值。
 */
inline __device__ float atomicMax(float *address, float val) {
  // atomicCAS 只支持整数类型
  int* address_as_i = (int*)address;
  // 读取当前存储在 address 处的整数值，并将其存储在变量 old 中。
  int old = *address_as_i;
  // 初始化一个假设值 assumed 为 0
  int assumed = 0;
  /**
   * 这个 do-while 循环是实现原子操作的核心部分。
   * 循环的目的是确保在多线程环境中，对 address 处的值进行原子更新。具体步骤如下：
   * 'assumed = old;' : 将假设值设置为当前值。
   * 'old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed)))); : 调用 atomicCAS 函数进行原子比较和交换操作。
   *  atomicCAS 会比较 address_as_i 处的值和 assumed 值，
   * 如果相等，则将 address_as_i 处的值更新为 __float_as_int(fmaxf(val, __int_as_float(assumed)))。
   * fmaxf 函数用于比较 val 和 assumed 的浮点数值，并返回最大值。
   * __float_as_int 和 __int_as_float 是CUDA提供的内置函数，用于在浮点数和整数之间进行转换。
   * } while (old != assumed);：如果 old 和 assumed 不相等，说明在比较和交换操作过程中，address_as_i 处的值被其他线程修改了，因此需要重新进行比较和交换操作。
   * */
  do {
    assumed = old;
    // 使用原子比较和交换操作（atomicCAS）来比较和更新值。
    // 如果当前值等于假设值，则将值更新为输入值和当前值中的最大值。
    old = atomicCAS(address_as_i, assumed,  __float_as_int(fmaxf(val, __int_as_float(assumed))));
  } while (old != assumed);
  // 将更新后的整数值转换回浮点数并返回。
  return __int_as_float(old);
}

/**
 * 在GPU设备上执行原子最小值操作。
 *
 * 该函数用于比较并更新指定地址上的浮点数，将其更新为输入值和当前值中的最小值。
 * 这是一个原子操作，确保在多线程环境下不会出现竞争条件。
 *
 * @param address 指向要比较和更新的浮点数的地址。
 * @param val 要与当前值比较的最小值。
 * @return 返回更新前的旧值。
 */
inline __device__ float atomicMin(float *address, float val) {
  // 将浮点数地址转换为整数地址，以便进行原子比较和交换操作。
  int* address_as_i = (int*)address;
  int old = *address_as_i;
  int assumed = 0;
  do {
    assumed = old;
    // 使用原子比较和交换操作（atomicCAS）来比较和更新值。
    // 如果当前值等于假设值，则将值更新为输入值和当前值中的最小值。
    old = atomicCAS(address_as_i, assumed,  __float_as_int(fminf(val, __int_as_float(assumed))));
  } while (old != assumed);
  // 将更新后的整数值转换回浮点数并返回。
  return __int_as_float(old);
}

// get max and min per tensor
// use block shared memory reduce，即reduce v4，唯一区别只是由reduce_sum变成了reduce_max_min
/**
 * 在GPU上实现针对每个张量的的最大值和最小值计算的并行化减少操作。
 *
 * 该函数旨在高效地计算给定输入数据集中的最大值和最小值。它利用了CUDA的并行计算能力，
 * 通过块级和网格级并行化来处理大量数据。
 *
 * @tparam T 输入数据的数据类型。
 * @param input_ptr 输入数据的指针。
 * @param nums 输入数据的总数量。
 * @param max_ptr 存储计算出的最大值的指针。
 * @param min_ptr 存储计算出的最小值的指针。
 * @param channel 输入数据的通道数。
 * @param HW 输入数据的高度和宽度的乘积。
 */
template<typename T>
__global__ void ReduceMaxMinPerTensor(const T* input_ptr, const int nums, T* max_ptr,
                                      T* min_ptr, const int channel, const int HW) {
  // 使用动态共享内存来存储每个线程块的最大值和最小值。
  // dyn shared memory
  extern __shared__ unsigned char shared_max_min_memory[];
  T* shared_max = reinterpret_cast<T*>(shared_max_min_memory);
  T* shared_min = shared_max + blockDim.x;

  // 计算总线程数。
  int total_thread_num = blockDim.x * gridDim.x;

  int tid = threadIdx.x;
  int gid = blockDim.x * blockIdx.x + tid;

  // 初始化每个线程的共享内存最大值为最小浮点数，最小值为最大浮点数。
  shared_max[tid] = FLT_MIN;
  shared_min[tid] = FLT_MAX;

  // 每个线程遍历分配给它的数据部分，更新共享内存中的最大值和最小值。
  // 1. block数量可能无法覆盖总数据量，先以total_thread_num把block和block范围外的数据给比较一遍
  for (int i = gid; i < nums; i += total_thread_num) {
    shared_max[tid] = max(shared_max[tid], input_ptr[i]);
    shared_min[tid] = min(shared_min[tid], input_ptr[i]);
  }

  // 等待所有线程完成其迭代。
  __syncthreads();


  // 2. 至此，所有block已经覆盖总数据量，于是开始在block内部先比较大小，又称intra-block范围的比较
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s && gid < nums) {
      shared_max[tid] = max(shared_max[tid], shared_max[tid + s]);
      shared_min[tid] = min(shared_min[tid], shared_min[tid + s]);
    }
    __syncthreads();
  }

  // 如果线程ID为0，则该线程负责将块的最大值和最小值更新到全局最大值和最小值存储位置。
  // 3. 最后，每个block里面的shared mem的0号位置都保存了block内部的最大最小值，此时使用atomic对所有block来进行比较
  if (tid == 0) {
    atomicMax(max_ptr, shared_max[0]);
    atomicMin(min_ptr, shared_min[0]);
  }
}

// get max and min per channel
/**
 * 在每个通道上计算最大值和最小值的CUDA内核函数。
 *
 * @tparam T 输入数据和输出结果的数据类型。
 * @param input_ptr 输入数据的指针。
 * @param nums 输入数据的总数量。
 * @param max_ptr 存储每个通道最大值的输出数组指针。
 * @param min_ptr 存储每个通道最小值的输出数组指针。
 * @param num_channels 输入数据的通道数量。
 * @param HW 每个通道的高度和宽度之和（H + W）。
 */
template<typename T>
__global__ void ReduceMaxMinPerChannel(const T* input_ptr, const int nums,
                                       T* max_ptr, T* min_ptr, const int num_channels, const int HW) {
  // 使用共享内存来存储每个线程块的最大值和最小值。
  extern __shared__ unsigned char shared_max_min_memory[];
  // 动态smem需要如下这样去强转成我们的计算类型，以及分配每块的大小.
  T* shared_max = reinterpret_cast<T*>(shared_max_min_memory);
  T* shared_min = shared_max + blockDim.x;

  // 当前处理的通道号。
  // block id represent channel id, if block nums < channel nums or thread nums < HW
  int cur_channel = blockIdx.x;
  // 线程在块中的ID。
  int tid = threadIdx.x;
  // 全局线程ID，用于计算全局索引。
  int gid = blockIdx.x * blockDim.x + tid;

  // 遍历所有通道，处理每个通道的最大值和最小值。
  // get min/max of each channel
  while (cur_channel < num_channels) {
    // 初始化每个线程的共享内存最大值为最小浮点数，最小值为最大浮点数。
    shared_max[tid] = FLT_MIN;
    shared_min[tid] = FLT_MAX;

    // 计算当前线程处理的数据在全局数组中的起始和结束索引。
    // index表示每个线程所取的元素位置，位于第cur channel的第tid偏移
    int index = (HW * cur_channel) + tid;
    int end = HW * (cur_channel + 1);

    // 遍历当前通道的数据，更新共享内存中的最大值和最小值。
    // 确定好了index，其他与reduceMinMaxPerTensor差不多
    // 适应thread per block < 每个channel里面的元素数量
    while (index < end && index < nums) {
      shared_max[tid] = max(shared_max[tid], input_ptr[index]);
      shared_min[tid] = min(shared_min[tid], input_ptr[index]);
      index += blockDim.x;
    }

    // 线程块内同步，确保所有线程都完成了数据加载。
    __syncthreads();

    // 使用并行归约算法，将块中的最大值和最小值合并到线程0的共享内存中。
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (tid < s) {
        shared_max[tid] = max(shared_max[tid], shared_max[tid + s]);
        shared_min[tid] = min(shared_min[tid], shared_min[tid + s]);
      }
      // 线程块内同步，确保所有线程都完成了当前的合并操作。
      __syncthreads();
    }

    // 线程0将块中的最大值和最小值写入全局输出数组。
    if (tid == 0) {
      atomicMax(&max_ptr[cur_channel], shared_max[0]);
      atomicMin(&min_ptr[cur_channel], shared_min[0]);
    }
    // 移动到下一个通道，以适应num_channels > gridDim.x的情况。
    cur_channel += gridDim.x;
  }
}

// Note: performance will be low if use cpu function
// why? min max ptr locate GPU memory ,not CPU memory.Because CPU funtion requests CPU memory data, so need
// copy max min ptr from GPU memory to CPU memory, which will incur some overhead!
// in addition, next kernel will run on GPU, at that time, scale and zeropoint will be copied from host to device,which will incur overhead, too
/**
 * @brief 计算用于Symmetric量化后的缩放因子和零点。
 *
 * @tparam T 数据类型。
 * @param max_ptr 存储最大值的指针。
 * @param min_ptr 存储最小值的指针。
 * @param nums 总的元素数量。
 * @param quantization_bit 量化位数。
 * @param scale 存储量化缩放因子的指针。
 * @param zero_point 存储量化零点的指针。
 */
template<typename T>
__global__ void GetScaleAndZPSymmetric(const T* max_ptr, const T* min_ptr,
                                       const int nums, const double quantization_bit,
                                       T* scale, T* zero_point) {
  // 线程索引和全局索引。
  int tid = threadIdx.x;
  int gid = blockDim.x * blockIdx.x + tid;
  // 遍历所有元素。
  while (gid < nums) {
    // 计算绝对值最大的权重。
    T weight_max = max(fabs(max_ptr[gid]), fabs(min_ptr[gid]));
    // 计算缩放因子的分母。
    T denominator = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;
    // 计算缩放因子并设置零点。
    scale[gid] = weight_max / denominator;
    zero_point[gid] = 0;
    // 更新全局索引。
    gid += gridDim.x * blockDim.x;
  }
}

/**
 * @brief 计算用于Asymmetric量化后的缩放因子和零点。
 *
 * @tparam T 数据类型。
 * @param max_ptr 存储最大值的指针。
 * @param min_ptr 存储最小值的指针。
 * @param nums 总的元素数量。
 * @param quantization_bit 量化位数。
 * @param scale 存储量化缩放因子的指针。
 * @param zero_point 存储量化零点的指针。
 */
template<typename T>
__global__ void GetScaleAndZPAsymmetric(const T* max_ptr, const T* min_ptr, const int nums,
                                        const double quantization_bit, T* scale, T* zero_point) {
  // 线程索引和全局索引。
  int tid = threadIdx.x;
  int gid = (blockDim.x * blockIdx.x) + tid;
  // 遍历所有元素。
  while (gid < nums) {
    // 计算缩放因子的分母。
    T denominator = static_cast<T>(pow(2.0, quantization_bit)) - 1;
    // 计算最小值的负值。
    T min = -min_ptr[gid];
    // 计算缩放因子。
    T s = (max_ptr[gid] - min) / denominator;
    scale[gid] = s;
    // 计算零点并处理浮点数到整数的转换。
    zero_point[gid] = -1 * std::nearbyint(min / s) - static_cast<T>(pow(2.0, quantization_bit - 1));
    // 更新全局索引。
    gid += gridDim.x * blockDim.x;
  }
}

/**
 * @brief 对输入数据进行通道内Symmetric量化。
 *
 * @tparam T 数据类型。
 * @param in_ptr 存储输入数据的指针。
 * @param scale_ptr 存储缩放因子的指针。
 * @param nums 总的元素数量。
 * @param quantization_bit 量化位数。
 * @param out_ptr 存储量化后数据的指针。
 * @param scale_size 缩放因子数组的大小。
 * @param HW 每个通道的元素数量（H * W）。
 */
template<typename T>
__global__ void QuantizePerChannelSymmetric(const T* in_ptr, const T* scale_ptr, const int nums,
                                            const double quantization_bit, T* out_ptr,
                                            const int scale_size, const int HW) {
  // 线程索引和全局索引。
  int gid = blockDim.x * blockIdx.x + threadIdx.x;
  int step = gridDim.x * blockDim.x;
  // 定义量化范围的上下界。
  T upper_bound = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;
  T lower_bound = -upper_bound - 1;
  // 遍历所有元素。
  while (gid < nums) {
    // 计算当前元素所属的通道索引。
    int channel_index = gid / HW;
    // 确保scale索引不超过数组大小。
    int scale_idx = min(scale_size - 1, channel_index);
    // 获取当前通道的缩放因子。
    T scale = scale_ptr[scale_idx];

    // 量化输入值并限制在合法范围内。
    T out = std::nearbyint(in_ptr[gid] / scale);
    out = out > upper_bound ? upper_bound : out;
    out = out < lower_bound ? lower_bound : out;
    out_ptr[gid] = out;

    // 更新全局索引。
    gid += step;
  }
}

/**
 * @brief 对输入数据进行通道内Asymmetric量化。
 *
 * @tparam T 数据类型。
 * @param in_ptr 存储输入数据的指针。
 * @param scale_ptr 存储缩放因子的指针。
 * @param zero_point_ptr 存储零点的指针。
 * @param scale_size 缩放因子数组的大小。
 * @param nums 总的元素数量。
 * @param HW 每个通道的元素数量（H * W）。
 * @param quantization_bit 量化位数。
 * @param out_ptr 存储量化后数据的指针。
 */
template<typename T>
__global__ void QuantizePerChannelAsymmetric(const T* in_ptr, const T* scale_ptr, const T* zero_point_ptr,
                                             const int scale_size, const int nums,
                                             const int HW, const double quantization_bit,
                                             T* out_ptr) {
  // 线程索引和全局索引。
  int gid = (blockDim.x * blockIdx.x) + threadIdx.x;
  int step = gridDim.x * blockDim.x;
  // 定义量化范围的上下界。
  T upper_bound = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;
  T lower_bound = -upper_bound - 1;

  while (gid < nums) {
    // 计算当前元素所属的通道索引。
    int channel_index = gid / HW;
    // 确保scale和zero_point索引不超过数组大小。
    int scale_idx = min(scale_size - 1, channel_index);

    // 获取当前通道的缩放因子和零点。
    T scale = scale_ptr[scale_idx];
    T zero_point = zero_point_ptr[scale_idx];

    // 量化输入值并限制在合法范围内。
    T out = std::nearbyint(in_ptr[gid] / scale + zero_point);
    out = out > upper_bound ? upper_bound : out;
    out = out < lower_bound ? lower_bound : out;
    out_ptr[gid] = out;

    // 更新全局索引。
    gid += step;
  }
}

// element wise operation
/**
 * @brief 对输入数据进行按张量均匀量化处理。
 *
 * 该函数执行按张量均匀量化的计算，适用于Symmetric量化方式。
 * 它通过将输入数据缩放到指定的量化位数，然后将其转换为整数类型，以减少模型的计算量和存储空间。
 *
 * @tparam T 输入和输出数据的类型。
 * @param in_ptr 输入数据的指针。
 * @param scale_ptr 缩放因子的指针。
 * @param nums 输入数据的数量。
 * @param quantization_bit 量化位数。
 * @param out_ptr 输出数据的指针。
 * @param channel 输入数据的通道数。
 * @param HW 输入数据的高和宽的乘积。
 */
template<typename T>
__global__ void QuantizePerTensorSymmetric(const T* in_ptr, const T* scale_ptr,
                                           const int nums, const double quantization_bit, T* out_ptr, const int channel, const int HW) {
  // 计算全局线程ID和步长。
  int gid = blockDim.x * blockIdx.x + threadIdx.x;
  int step = gridDim.x * blockDim.x;

  // 计算量化的上限和下限。
  T upper_bound = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;
  T lower_bound = -upper_bound - 1;
  T scale = *scale_ptr;
  if (gid==0) printf("scaleGPU is %f\n", scale);
  while (gid < nums) {
    // 根据缩放因子对输入数据进行量化。
    // per tensor quant和per channel quant的最大区别在于这里不用去根据channel ID取对应的scale，而是整个tensor共用一个scale
    T out = gpuNearbyint(in_ptr[gid] / scale);
    // 确保量化结果在有效范围内。
    out = out > upper_bound ? upper_bound : out;
    out = out < lower_bound ? lower_bound : out;
    // 将量化结果写入输出数组。
    out_ptr[gid] = out;

    // 更新全局线程ID。
    gid += step;
  }
}

/**
 * @brief 对输入数据进行按张量非均匀量化处理。
 *
 * 该函数执行按张量非均匀量化的计算，适用于Asymmetric量化方式。
 * 它考虑了零点的补偿，通过将输入数据先缩放，然后加上零点偏移，最后进行量化，以更精确地逼近原始浮点数。
 *
 * @tparam T 输入和输出数据的类型。
 * @param in_ptr 输入数据的指针。
 * @param scale_ptr 缩放因子的指针。
 * @param zero_point_ptr 零点的指针。
 * @param nums 输入数据的数量。
 * @param quantization_bit 量化位数。
 * @param out_ptr 输出数据的指针。
 * @param channel 输入数据的通道数。
 * @param HW 输入数据的高和宽的乘积。
 */
template<typename T>
__global__ void QuantizePerTensorAsymmetric(const T* in_ptr, const T* scale_ptr, const T* zero_point_ptr,
                                            const int nums, const double quantization_bit, T* out_ptr,
                                            const int channel, const int HW) {
  // 计算全局线程ID和步长。
  int gid = blockDim.x * blockIdx.x + threadIdx.x;
  int step = gridDim.x * blockDim.x;

  // 计算量化的上限和下限。
  T upper_bound = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;
  T lower_bound = -upper_bound - 1;
  T scale = *scale_ptr;
  T zero_point = *zero_point_ptr;
  while (gid < nums) {
    // 根据缩放因子和零点对输入数据进行量化。
    T out = nearbyint(in_ptr[gid] / scale + zero_point);
    // 确保量化结果在有效范围内。
    out = out > upper_bound ? upper_bound : out;
    out = out < lower_bound ? lower_bound : out;
    // 将量化结果写入输出数组。
    out_ptr[gid] = out;

    // 更新全局线程ID。
    gid += step;
  }
}

// use macro to reduce redundant code
#define LAUNCH_GPU_KERNEL(GetMinMaxFunc, QuantFunc, scale_size, channel, HW) \
    cudaMalloc((void **)&d_scale, scale_size * sizeof(float)); \
    cudaMalloc((void **)&d_zeropoint, scale_size * sizeof(float)); \
    cudaMalloc((void **)&d_max, scale_size * sizeof(float)); \
    cudaMalloc((void **)&d_min, scale_size * sizeof(float)); \
    cudaEvent_t start, stop; \
    cudaEventCreate(&start); \
    cudaEventCreate(&stop); \
    cudaEventRecord(start); \
    GetMinMaxFunc<float><<<gridSize, blockSize, blockSize * 2 * sizeof(float), 0>>>(d_input, nums, d_max, d_min, channel, HW);  \
    GetScaleAndZPSymmetric<float><<<1, blockSize>>>(d_max, d_min, channel, quantization_bit, d_scale, d_zeropoint); \
    QuantFunc<float><<<gridSize, blockSize>>>(d_input, d_scale, nums, quantization_bit, d_output, channel, HW); \
    cudaEventRecord(stop); \
    cudaEventSynchronize(stop); \
    cudaEventElapsedTime(&milliseconds, start, stop);

int main() {
  float milliseconds = 0;
  constexpr int nums = 400 * 20 * 10;
  constexpr int HW = 20 * 10;
  constexpr int channel = 400;
  constexpr int quantization_bit = 8;
  float* input = (float*) malloc(sizeof(float) * nums);
  float cpu_min = FLT_MAX;
  float cpu_max = FLT_MIN;
  for(int i = 0; i < nums; i++) {
    // generate float input inside [-1, 1],[-3,3]
    input[i] = -3 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/6));
    cpu_min = std::min(input[i], cpu_min);
    cpu_max = std::max(input[i], cpu_max);
  }
  // printf("per tensor min max cpu are  %f, %f\n", cpu_min, cpu_max);
  auto* output = (float*) malloc(sizeof(float) * nums);
  float *d_input, *d_output;
  cudaMalloc((void **)&d_input, nums * sizeof(float));
  cudaMalloc((void **)&d_output, nums * sizeof(float));
  cudaMemcpy(d_input, input, sizeof(float) * nums, cudaMemcpyHostToDevice);
  // block and thread config
  cudaDeviceProp deviceProp{};
  cudaGetDeviceProperties(&deviceProp, 0);
  int maxblocks = deviceProp.maxGridSize[0];
  int blockSize = 256;
  int gridSize = std::min<int>((nums + blockSize - 1) / blockSize,  std::min<int>(maxblocks, channel));
  printf("gridsize blocksize are  %d, %d\n", gridSize, blockSize);
  float *d_scale, *d_zeropoint, *d_max, *d_min;
  // per tensor, scale and zp shape both are 1
  // switch to per tensor
  bool per_tensor_quantize = false;
  if(per_tensor_quantize) {
    //cudaMalloc((void **)&d_scale, 1 * sizeof(float));
    //cudaMalloc((void **)&d_zeropoint, 1 * sizeof(float));
    //cudaMalloc((void **)&d_max, 1 * sizeof(float));
    //cudaMalloc((void **)&d_min, 1 * sizeof(float));
    //cudaEvent_t start, stop;
    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);
    //cudaEventRecord(start);
    //ReduceMaxMinPerTensor<float><<<gridSize, blockSize, blockSize * 2 * sizeof(float), 0>>>(d_input, nums, d_max, d_min);
    //GetScaleAndZPSymmetric<float><<<1, 1>>>(d_max, d_min, nums, quantization_bit, d_scale, d_zeropoint);//scale only shape 1
    //QuantizePerTensorSymmetric<float><<<gridSize, blockSize>>>(d_input, d_scale, nums, quantization_bit, d_output);
    //cudaEventRecord(stop);
    //cudaEventSynchronize(stop);
    //cudaEventElapsedTime(&milliseconds, start, stop);
    LAUNCH_GPU_KERNEL(ReduceMaxMinPerTensor, QuantizePerTensorSymmetric, 1, nums, HW);
  } else {
  // switch to per channel
    LAUNCH_GPU_KERNEL(ReduceMaxMinPerChannel, QuantizePerChannelSymmetric, channel, channel, HW);
    //cudaMalloc((void **)&d_scale, channel * sizeof(float));
    //cudaMalloc((void **)&d_zeropoint, channel * sizeof(float));
    //cudaMalloc((void **)&d_max, channel * sizeof(float));
    //cudaMalloc((void **)&d_min, channel * sizeof(float));
    //cudaEvent_t start, stop;
    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);
    //cudaEventRecord(start);
    //ReduceMaxMinPerChannel<float><<<gridSize, blockSize, blockSize * 2 * sizeof(float), 0>>>(d_input, nums, d_max, d_min, channel, HW);
    //GetScaleAndZPSymmetric<float><<<1, blockSize>>>(d_max, d_min, channel, quantization_bit, d_scale, d_zeropoint);
    //QuantizePerChannelSymmetric<float><<<gridSize, blockSize>>>(d_input, d_scale, nums, quantization_bit, d_output, channel, HW);
    //cudaEventRecord(stop);
    //cudaEventSynchronize(stop);
    //cudaEventElapsedTime(&milliseconds, start, stop);
  }

  cudaMemcpy(output, d_output, sizeof(float) * nums, cudaMemcpyDeviceToHost);
  // (per tensor) get CPU output to validate GPU result is right or not
  auto* CPUOutput= (float*) malloc(sizeof(float) * nums);
  if(per_tensor_quantize) {
    auto* scale = (float*) malloc(sizeof(float) * 1);
    auto* zeropoint = (float*) malloc(sizeof(float) * 1);
    GenScalePerTensorSymmetricCPU<float>(input, quantization_bit, nums, scale, zeropoint);
    QuantizationPerTensorSymmetricCPU<float>(input, *scale, quantization_bit, nums, CPUOutput);
    free(scale);
    free(zeropoint);
  } else {
    auto* scale = (float*) malloc(sizeof(float) * channel);
    auto* zero_point = (float*) malloc(sizeof(float) * channel);
    GenScalePerChannelSymmetricCPU<float>(input, quantization_bit, HW, channel, nums, scale, zero_point);
    QuantizationPerChannelSymmetricCPU<float>(input, scale, quantization_bit, HW, nums, CPUOutput);
    free(scale);
    free(zero_point);
  }
  if (CheckResult(output, CPUOutput, nums)) {
    printf("the ans is right");
  } else {
    printf("the ans is wrong\n");
    printf("first two CPU output are %f, %f\n", CPUOutput[0], CPUOutput[1]);
    printf("first two output are %f, %f\n", output[0], output[1]);
  }
  printf("Quantize kernel latency = %f ms\n", milliseconds);
  free(input);
  free(output);
  free(CPUOutput);
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_scale);
  cudaFree(d_zeropoint);
  cudaFree(d_max);
  cudaFree(d_min);
}
