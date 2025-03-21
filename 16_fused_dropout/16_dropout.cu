#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>

#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

// 自定义向量化类型，主要用于VecType_u8
template<typename T, int VecSize>
struct alignas(sizeof(T) * VecSize) VectorType {
    T val[VecSize];
};

// 随机数生成器
template<typename T>
struct uniform_distribution {
    __device__ T operator()(curandStatePhilox4_32_10_t *state) {
        return static_cast<T>(curand_uniform(state));
    }

    static constexpr int Count = 1;
};

template<>
struct uniform_distribution<float> {
    __device__ float4 operator()(curandStatePhilox4_32_10_t *state) {
        return curand_uniform4(state);
    }

    static constexpr int Count = 4;
};

// 计算输出和mask的最小逻辑
template<typename T>
struct DstMaskFunctor {
    const float prob_;
    const bool is_upscale_in_train_;
    float inv_prob;
    __device__ DstMaskFunctor(const float prob, const bool is_upscale_in_train)
        : prob_(prob), is_upscale_in_train_(is_upscale_in_train) {
        inv_prob = 1.0f / (1 - prob_);
    }

    /**
   * @brief 根据随机数和概率对数据进行处理的运算符重载函数。
   *
   * 此函数用于根据给定的概率对数据进行处理：如果随机数小于给定的概率prob_，
   * 则在目标数组中对应的元素置为0，并在后续位置(dst前vec size位置放处理后的数据，后vec size放mask值)记录这个0；
   * 如果随机数大于等于概率，则根据是否处于训练阶段决定是放大还是保持原样，并在后续位置记录1。
   * 这种处理方式常用于数据增强技术中，以增加模型的泛化能力。
   *
   * @tparam T 数据类型。
   * @param dst 输出数组，用于存放处理后的数据 + mask。
   * @param src_val 输入数组，包含原始数据。
   * @param rand 随机数数组，用于决定每个数据点如何处理。
     */
    __device__ void operator()(T *dst, const T *src_val, const T *rand) {
        // Count定义了处理数据的个数。
        static constexpr int Count = uniform_distribution<T>::Count;
        for (int i = 0; i < Count; i++) {
            // 根据随机数和概率决定数据处理方式。
            if (rand[i] < prob_) {
                // 如果随机数小于概率，则将数据置为0，并在后续位置记录这个0。
                dst[i] = static_cast<T>(0);
                dst[i + Count] = dst[i];
            } else {
                // 如果随机数大于等于概率，则根据训练阶段决定数据处理方式。
                dst[i] = is_upscale_in_train_
                             ? static_cast<T>(src_val[i] * inv_prob)
                             : static_cast<T>(src_val[i]);
                // 在后续位置记录1，表示数据经过处理。
                dst[i + Count] = static_cast<T>(1);
            }
        }
    }
};

/**
 * @brief 生成对应的掩码，并使用向量化的dropout算法对输入数据进行处理。
 *
 * 此函数在GPU上执行，利用CUDA的并行计算能力，对大规模数据集进行dropout操作。
 * Dropout是一种神经网络训练中的正则化方法，本函数实现了其向量化版本，以提高计算效率。
 *
 * @tparam T 输入数据的类型。
 * @tparam MaskType 掩码数据的类型。
 * @param n 输入数据的总元素数量。
 * @param seed 随机数生成的种子。
 * @param dropout_prob Dropout的概率，即保留每个元素的概率。
 * @param src 输入数据的指针。
 * @param mask 生成的掩码数据的指针。
 * @param dst 处理后的数据输出指针。
 * @param is_upscale_in_train 在训练过程中是否放大保留元素的值。即是否需要1/(1-p)。
 * @param increment 随机数生成的增量。
 * @param main_offset 主要的偏移量--->可以向量化的数据量。
 */
template<typename T, typename MaskType>
__global__ void VectorizedDstMask(const size_t n,
                                  int seed,
                                  const float dropout_prob,
                                  const T *src,
                                  MaskType *mask,
                                  T *dst,
                                  bool is_upscale_in_train,
                                  int increment,
                                  int main_offset) {
    int thread_idx = threadIdx.x;
    int threads_per_block = blockDim.x;
    int block_idx = blockIdx.x;
    int block_nums = gridDim.x;

    int block_offset = block_idx * threads_per_block;
    int gtid = block_offset + thread_idx;
    constexpr int VecSize = uniform_distribution<float>::Count;
    // 所有block在一次迭代中的数据处理总量
    int stride = block_nums * threads_per_block * VecSize;
    // 初始化随机数状态
    // 声明一个Philox4_32_10类型的随机数状态对象
    curandStatePhilox4_32_10_t state;
    // 初始化随机数生成器的状态
    // curand_init()函数用于初始化随机数生成器的状态。
    // 参数1: seed - 随机数生成的种子，用于确定随机数序列的起点。
    // 参数2: block_offset - 每个block的偏移量，用于在多线程环境下产生不同的随机数序列。
    // 参数3: increment - 每个线程的增量，进一步区分不同线程产生的随机数序列。
    // 参数4: &state - 随机数生成器的状态指针，用于保存生成器的内部状态。
    curand_init(seed, gtid, increment, &state);
    // 声明相关寄存器，暂存输出数据、随机数、mask
    T dst_mask[VecSize * 2]; // 0 ~ VecSize -1 : dst;VecSize ~ 2 * VecSize - 1: mask
    float rands[VecSize];
    MaskType mask_result[VecSize];
    // 初始化生成随机数的functor、计算mask和输出结果的functor
    using Rand = uniform_distribution<float>;
    auto dst_functor =
            DstMaskFunctor<T>(dropout_prob, is_upscale_in_train);

    using VecType = float4;
    using VecType_u8 = VectorType<MaskType, VecSize>;
    VecType vec_temp_input;
    // 可以向量化的部分
    int start = block_offset * VecSize;
    for (; start < main_offset; start += stride) {
        // 取出数据
        int thread_offset = thread_idx;
        const VecType *vec_input = reinterpret_cast<const VecType *>(src + start);
        vec_temp_input = vec_input[thread_offset];
        auto random_tuple = Rand()(&state);
        for (int i = 0; i < VecSize; i++) {
            dst_mask[i] = *(reinterpret_cast<T *>(&vec_temp_input) + i);
            rands[i] = static_cast<float>((&random_tuple.x)[i]);
        }
        // 算出数据
        dst_functor(&dst_mask[0], &dst_mask[0], &rands[0]);
        // 写回数据
        T *res = dst + start;
        VecType *vec_dst_output = reinterpret_cast<VecType *>(res);
        vec_dst_output[thread_offset] = *(reinterpret_cast<VecType *>(&dst_mask[0]));

        for (int i = 0; i < VecSize; i++) {
            mask_result[i] = static_cast<MaskType>(dst_mask[i + VecSize]);
        }

        MaskType *mask_res = mask + start;
        VecType_u8 *vec_mask_output = reinterpret_cast<VecType_u8 *>(mask_res);
        vec_mask_output[thread_offset] =
                *(reinterpret_cast<VecType_u8 *>(mask_result));
    }
    // 不可向量化的部分
    int remain = n - start;
    if (remain > 0) {
        // 取出数据
        int thread_offset = thread_idx * VecSize;
        const T *src_remain = src + start;
        auto random_tuple = Rand()(&state);
        for (int i = 0; i < VecSize; i++) {
            if (i + thread_offset < remain) {
                dst_mask[i] = src_remain[thread_offset + i];
            }
            rands[i] = static_cast<float>((&random_tuple.x)[i]);
        }
        // 算出数据
        dst_functor(&dst_mask[0], &dst_mask[0], &rands[0]);
        // 写回数据
        T *res = dst + start;
        MaskType *mask_res = mask + start;
        for (int i = 0; i < VecSize; i++) {
            if ((thread_offset + i) < remain) {
                res[thread_offset + i] = dst_mask[i];
                mask_result[i] = static_cast<MaskType>(dst_mask[i + VecSize]);
                mask_res[thread_offset + i] = mask_result[i];
            }
        }
    }
}

/**
 * @brief 实现dropout操作的内核函数。
 *
 * Dropout是一种神经网络训练中的正则化方法，它在训练过程中以一定概率随机关闭神经元，
 * 以防止过拟合。本函数在训练阶段应用dropout，并在推理阶段简单地复制输入到输出。
 *
 * @tparam T 输入数据的类型。
 * @param is_test 指示当前是否为测试（推理）阶段的布尔值。
 * @param is_upscale_in_train 在训练过程中是否放大剩余神经元输出的布尔值。
 * @param num_eles 需要进行dropout操作的元素数量。
 * @param dropout_prob 每个元素被dropout的概率。
 * @param seed_val 随机数生成的种子值。
 * @param x_data 输入数据的设备内存地址。
 * @param mask_data 用于dropout操作的掩码数据的设备内存地址。
 * @param y_data 经过dropout操作后的输出数据的设备内存地址。
 */
template<typename T>
void DropoutKernel(const bool is_test,
                   const bool is_upscale_in_train,
                   const size_t num_eles,
                   const float dropout_prob,
                   const int seed_val,
                   const float *x_data,
                   uint8_t *mask_data,
                   float *y_data) {
    // 1. 训练: dropout最多用的场景，丢弃某些神经元
    if (!is_test) {
        if (dropout_prob == 1.0f) {
            cudaMemset(y_data, 0, num_eles);
            cudaMemset(mask_data, 0, num_eles);
            return;
        }

        // 每个线程负责生成4个随机数
        constexpr int RandVecSize = uniform_distribution<float>::Count;

        size_t num_blocks = 2;
        size_t block_size = 256;
        dim3 grid(num_blocks);
        dim3 block(block_size);

        int seed_data = seed_val;
        int increment = 0;
        // 可向量化读写的数据量
        // num_blocks * block_size * 4 = 开启的线程数一次能够读多少fp32的元素
        // 总元素量num_eles / (num_blocks * block_size * 4) 得到循环次数
        // num_eles / (num_blocks * block_size * 4) * (num_blocks * block_size * 4) 得到可以向量化的数据量。
        int main_offset =
                num_eles / (num_blocks * block_size * 4) * (num_blocks * block_size * 4);

        VectorizedDstMask<T, uint8_t><<<grid, block>>>(num_eles,
                                                       seed_data,
                                                       dropout_prob,
                                                       x_data,
                                                       mask_data,
                                                       y_data,
                                                       is_upscale_in_train,
                                                       increment,
                                                       main_offset);
    } else {
        // 2. 推理场景，output=input
        cudaMemcpy(y_data, x_data, num_eles, cudaMemcpyDeviceToDevice);
    }
}

int main() {
    // 512 * 4 + 2。在fp32下，使用float4进行读取，要读512个float4，还剩两个fp32类型的元素
    constexpr size_t num_eles = 2050;

    auto *x = (float *) malloc(num_eles * sizeof(float));
    float *d_x;
    CHECK(cudaMalloc((void **)&d_x, num_eles * sizeof(float)));

    auto *y = (float *) malloc(num_eles * sizeof(float));
    float *d_y;
    CHECK(cudaMalloc((void **)&d_y, num_eles * sizeof(float)));

    // 因为mask只是0或1，所以是uint8_t类型。用于记录每个元素是否被丢弃。
    auto *mask = (uint8_t *) malloc(num_eles * sizeof(uint8_t));
    uint8_t *d_mask;
    CHECK(cudaMalloc((void **)&d_mask, num_eles * sizeof(uint8_t)));

    for (int i = 0; i < num_eles; i++) {
        x[i] = 1;
    }

    CHECK(cudaMemcpy(d_x, x, num_eles * sizeof(float), cudaMemcpyHostToDevice));
    const bool is_test = false;
    const bool is_upscale_in_train = true;
    const float dropout_prob = 0.5;
    const int seed_val = 10000;
    DropoutKernel<float>(is_test,
                         is_upscale_in_train,
                         num_eles,
                         dropout_prob,
                         seed_val,
                         d_x,
                         d_mask,
                         d_y);
    CHECK(cudaMemcpy(y, d_y, num_eles * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(mask, d_mask, num_eles * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    // 打印最后位于可向量化和不可向量化边界的三个结果
    for (int i = num_eles - 3; i < num_eles; i++) {
        printf("[%d] y is %f\n", i, y[i]);
        printf("[%d] mask is %d\n", i, mask[i]);
    }
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_mask);
    free(x);
    free(y);
    free(mask);
}
