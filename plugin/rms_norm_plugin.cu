#include "rms_norm_plugin.h"
#include <cub/cub.cuh>
using namespace nvinfer1;

PluginFieldCollection    RMSNormPluginCreator::fc_ {};
std::vector<PluginField> RMSNormPluginCreator::attr_;


template<typename T, int n>
__global__ void RMSNorm(
    T * input, T * output, float epsilon, const T * gamma) {
  /*
  注意：这个计算结果都是以T类型进行储存，如果遇到fp16,精度会有问题
        如果遇到fp16精度,则gamma中间结果需要以float类型保存
  params input: 输入数据
  params gamma: 用于RMSNorm后相乘该对象
  params output: 最终输出结果
  */
  // 获取当前索引，与batch_索引
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  const int tx = threadIdx.x; 

  // Cast计算，更换类型
  T _x = input[idx];

  // printf("idx is %d, tx is %d, x is %f\n", idx, tx, _x);
  // 注意gamma只与最后一个维度有关，所以只和tx相关就行了
  T _a = static_cast<T>(gamma[tx]);
  // 求方差
  __shared__ T var;
  // Reduce操作
  //typedef cub::BlockReduce<T,n> BlockReduce;
  __shared__ typename cub::BlockReduce<T, n>::TempStorage temp;
  // 开始求方差了
  T var_temp = _x * _x;
  T & ref2 = var_temp;
  T temp_var = cub::BlockReduce<T,n>(temp).Sum(ref2);
  if (tx == 0) {
    var = temp_var / static_cast<T>(n);
  }
  __syncthreads();
  // 获取最终结果
  output[idx] = (_x) * static_cast<T>(
    rsqrtf(var + static_cast<T>(epsilon))
  ) * _a;
}

template <int VPT>
struct BytesToType;
// 
template <>
struct BytesToType<2>
{
    using type = uint16_t;
};
template <>
struct BytesToType<4>
{
    using type = uint32_t;
};
template <>
struct BytesToType<8>
{
    using type = uint64_t;
};
template <>
struct BytesToType<16>
{
    using type = float4;
};


template <int Bytes>
__device__ inline void copy(const void* local, void* data)
{
    using T = typename BytesToType<Bytes>::type;
// 
    const T* in = static_cast<const T*>(local);
    T* out = static_cast<T*>(data);
    *out = *in;
}


__device__ inline float2 operator + (const  float2 &a, const float2 & b) {
  float2 out{0.0f, 0.0f};
  // printf("a.x %f, b.x %f\t a.y %f, b.y %f\n", a.x, b.x, a.y, b.y);
  out.x = a.x + b.x;
  out.y = a.y + b.y;
  return out;
}


template<typename T, int n, int TPB, int VPT>
__global__ void RMSNormV2(
    const T * input, T * output, float epsilon, const T * gamma) {
  /*
  cub v2版，将输出结果用float储存，无论中间是否为half数据还是float32数据
  使用cub内置的内存管理进行数据拷贝，减少数据拷贝时间。
  params T: 数据类型，默认为fp32或者fp16
  param n: 数据长度，对于三维输入变量，一般为最后一个维度的值
  param VPT: 为16除以数据类型的个数，16为数据对齐的最小单位, 展开计算速度会快一些
  param TPB: 数据束个数，如果n < 32, 则取32；否则取 n / VPT的值
  params input: 输入数据
  params output: 最终输出结果
  params epsilon: 除标准差的时候加的一个小系数，防止分母为零
  params gamma: 用于LayerNorm后相乘该对象
  */
  const int idx = threadIdx.x * VPT + blockIdx.x * n;
  const int tx = threadIdx.x * VPT;
  // 准备本地数据
  T local_x[VPT], local_gamma[VPT];
  // 复制一个束的数据到local相关数据中，加快速度
  copy<sizeof(T) * VPT>(&input[idx], local_x);
  // 计算长度的倒数，用于后续求均值用(注意这里还是用float，中间精度float更好)
  const float r_length = float(1) / float(n);
  // 储存方差，同时完成
  float2 local_float2 {0.f};

#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    const float temp = r_length * (float)local_x[i];
    local_float2.x += temp * (float)local_x[i];
  }
  // 开始拷贝gamma与beta,注意，gamma与beta都是一维变量，只与tx有关，与idx无关
  copy<sizeof(T) * VPT>(&gamma[tx], local_gamma);
  // 利用blockReduce计算所有线程的均值和方差，均值与方差可以同时完成计算
  using BlockReduce = cub::BlockReduce<float2, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ float rsigma; // 1/标准差
  const float2 sum2 = BlockReduce(temp_storage).Reduce(local_float2, cub::Sum());

  // 获取最终结果
  if (threadIdx.x == 0) {

    rsigma = rsqrt(sum2.x + epsilon);
  }
  __syncthreads();

  // 展开循环体 - 计算最终的LayerNorm的值
#pragma unroll
  // printf("idx %d tx %d   mu %f gamma %f beta %f\n", idx, tx, mu, local_gamma[0], local_beta[0]);
  for (int i = 0; i < VPT; ++i) {
    local_x[i] = (float)local_gamma[i] * ((float)local_x[i] * rsigma);
  }
  // 将对应数据拷贝回output
  copy<sizeof(T) * VPT>(local_x, &output[idx]);
}


template __global__ void RMSNormV2<float, 256 ,64, 4>(const float*, float*, const float, const float*);
template __global__ void RMSNormV2<half, 256 ,32, 8>(const half*, half*, const float, const half *);
template __global__ void RMSNormV2<float, 768 ,192, 4>(const float*, float*, const float, const float*);
template __global__ void RMSNormV2<half, 768 , 96, 8>(const half*, half*, const float, const half *);
template __global__ void RMSNormV2<float, 4096 ,1024, 4>(const float*, float*, const float, const float*);
template __global__ void RMSNormV2<half, 4096 , 512, 8>(const half*, half*, const float, const half *);

int32_t RMSNormPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    int nBlock = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];
    int nValuePerBlock = inputDesc[0].dims.d[2];
    if (inputDesc[0].type == DataType::kFLOAT)
    {
        std::size_t nbytes = length_ * sizeof(float);
        void * temp1 {nullptr};
        checkRuntime(cudaMallocAsync(&temp1, nbytes, stream));
        checkRuntime(
            cudaMemcpyAsync(
                temp1, h_gamma.data(), nbytes, cudaMemcpyHostToDevice, stream));
        float * gamma = static_cast<float *>(temp1);

        const int VPT = 16 / sizeof(float);
        switch (nValuePerBlock)
        {
        case 256: { // 仅用于处理 nHiddenDimension 为 256 的情况SA
          const int TPB = 256 / VPT; 
          (RMSNormV2<float, 256, TPB, VPT>)<<<nBlock, TPB, 0, stream>>>(
              (float *)inputs[0], (float *)outputs[0],
              epsilon_, gamma
          );
          LOG_INFO("调用核函数完成, FLOAT32类型, length 256");
          break;
        }
        case 768: { // 仅用于处理 nHiddenDimension 为 768 的情况
          const int TPB = 768 / VPT; 
          (RMSNormV2<float, 768, TPB, VPT>)<<<nBlock, TPB, 0, stream>>>(
              (float *)inputs[0], (float *)outputs[0],
               epsilon_, gamma
          );
          LOG_INFO("调用核函数完成, FLOAT32类型, length 768");
          break;
        }
        case 4096: {
          const int TPB = 4096 / VPT; 
          (RMSNormV2<float, 4096, TPB, VPT>)<<<nBlock, TPB, 0, stream>>>(
              (float *)inputs[0], (float *)outputs[0],
               epsilon_, gamma
          );
          LOG_INFO("调用核函数完成, FLOAT32类型, length 4096");
          break;
        }
        case 32:
          (RMSNormV2<float, 32, 32, 1>)<<<nBlock, nValuePerBlock, 0, stream>>>(
              (float *)inputs[0], (float *)outputs[0],
              epsilon_, gamma
          );
          LOG_INFO("调用核函数完成, FLOAT32类型, length 8");
          break;
        default: // shoulf NOT be here
            LOG_ERROR("你输入的长度类型" + std::to_string(nValuePerBlock) + "暂不支持");
            LOG_ERROR("当前仅支持长度768, 256, 8三种");
            break;
        }
      cudaDeviceSynchronize();
      cudaFree(gamma);
    }
    else
    {
        std::size_t nbytes = length_ * sizeof(half);
        half * temp1 {nullptr};
        checkRuntime(cudaMallocAsync(&temp1, nbytes, stream));
        checkRuntime(
            cudaMemcpyAsync(
                temp1, h_gamma.data(), nbytes, cudaMemcpyHostToDevice, stream
            )
        );
        half * gamma = static_cast<half *>(temp1);
        const int VPT = 16 / sizeof(half);
        switch (nValuePerBlock)
        {
        case 256: { // 仅用于处理 nHiddenDimension 为 256 的情况
          const int TPB1 = 256 / VPT;
          (RMSNormV2<half, 256, TPB1, VPT>)<<<nBlock, TPB1, 0, stream>>>(
            (half *)inputs[0], (half *)outputs[0], epsilon_, gamma
          );
          LOG_INFO("调用核函数完成, FP16类型, length 256");
          break;
        }
        case 768: { // 仅用于处理 nHiddenDimension 为 768 的情况
          const int TPB2 = 768 / VPT;
          (RMSNormV2<half, 768, TPB2, VPT>)<<<nBlock, TPB2, 0, stream>>>(
              (half *)inputs[0], (half *)outputs[0], epsilon_, gamma
          );
          LOG_INFO("调用核函数完成, FP16类型, length 768");
          break;
        } 
        case 4096: {
          const int TPB2 = 4096 / VPT;
          (RMSNormV2<half, 4096, TPB2, VPT>)<<<nBlock, TPB2, 0, stream>>>(
              (half *)inputs[0], (half *)outputs[0], epsilon_, gamma
          );
          LOG_INFO("调用核函数完成, FP16类型, length 4096");
          break;
        } 
        case 32: // 仅用于处理 nHiddenDimension 为 32 的情况
            (RMSNormV2<half, 32, 32, 1>)<<<nBlock, nValuePerBlock, 0, stream>>>(
                (half *)inputs[0], (half *)outputs[0],
                epsilon_, gamma
            );
            LOG_INFO("调用核函数完成, FP16类型, length 8");
            break;

        default: // shoulf NOT be here
            LOG_ERROR("你输入的长度类型" + std::to_string(nValuePerBlock) + "暂不支持");
            LOG_ERROR("当前仅支持长度768, 256, 8三种");
            break;
        }
    }
    return 0;
}

REGISTER_TENSORRT_PLUGIN(RMSNormPluginCreator);


// 正式运行时注释掉，只为测试用
/*
void cpu_layer_norm(
  const float * inputs, float * output, float epsilon, const int n,
  const int batch, const  float * gamma, const  float * beta) {
  // CPU版的RMSNorm计算
  for (int b = 0; b < batch; ++b) {
    float sum = 0;
    float var = 0;
    int idx = 0;
    for (int i = 0; i < n; ++i) {
      idx = i + b * n;
      sum += inputs[idx];
      var += inputs[idx] * inputs[idx];
    }
    float mean = sum / n;
    var = var / n;
    float rstd = 1 / (sqrt(var - mean * mean) + epsilon);
    for (int i = 0; i < n; ++i) {
      idx = i + b * n;
      output[idx] = (inputs[idx] - mean) * rstd * gamma[i] + beta[i];
    }
  }
}


bool check_result(
  const float * h_output1, const float * h_output2,
  const int total, const float epsilon = 1e-5) {
  // 检查数据是否有问题
  for (int i = 0; i < total; ++i) {
    if (abs(h_output1[i] - h_output2[i]) > epsilon) {
      printf("max value is %f", abs(h_output1[i] - h_output1[i]));
      return false;
    }
  }
  return true;
}


int main() {
  const size_t n = 32;
  const int batch = 2;
  std::vector<float> h_input(n * batch);
	// float h_input[n * batch];
	// float h_gamma[n];
	// float h_beta[n];
  std::vector<float> h_gamma(n, 1);
  std::vector<float> h_beta(n, 0);
  for (int i = 0; i < n; ++i) {
    h_input[i] = i + 1;
    // h_gamma[i] = 1;
    // h_beta[i] = 0;
  }
  for (int b = 1; b < batch; ++b) {
    for (int i = 0; i < n; ++i) {
      h_input[i + b * n] = i + 1;
    }
  }
  // 打印输入
  // printf("input is :\n");
  // for (int b = 0; b < batch; ++b) {
  //   for (int i = 0; i < n; ++i) {
  //     printf("%.0f\t", h_input[i]);
  //   }
  //   printf("\n");
  // }
	size_t nbytes = n * sizeof(float);
	// float h_output[n * batch];
  std::vector<float> h_output(n * batch, 0);
	float * d_input, * d_output, * d_gamma, * d_beta;
	cudaMalloc(&d_input, nbytes * batch);
	cudaMalloc(&d_output, nbytes * batch);
	cudaMemcpy(d_input, h_input.data(),nbytes * batch, cudaMemcpyHostToDevice);

	cudaMalloc(&d_gamma, nbytes);
	cudaMalloc(&d_beta, nbytes);
	cudaMemcpy(d_gamma, h_gamma.data(), nbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_beta, h_beta.data(), nbytes, cudaMemcpyHostToDevice);

	d_input = static_cast<float *>(d_input);
	d_output = static_cast<float *>(d_output);
	d_gamma = static_cast<float *>(d_gamma);
	d_beta = static_cast<float *>(d_beta);
	// (RMSNorm<float, 32>)<<<batch, 32>>>(d_input, d_output, 0.01f, d_gamma, d_beta);
  const int VPT = 16 / sizeof(float);
  float epsilon = 1.0e-12;
	(RMSNormV2<float, n, n / VPT, VPT>)<<<batch, n / VPT>>>(
    d_input, d_output, epsilon, d_gamma, d_beta);
  cudaMemcpy(h_output.data(), d_output, nbytes * batch, cudaMemcpyDeviceToHost);
  // printf("output is :\n");
  // for (int b = 0; b < batch; ++b){
  //   for (int i = 0; i < n; ++i) {
  //     printf("res %f\t", h_output[i + b * n]);
  //   }
  //   printf("\n");
  // }

  // printf("\n==========================================================\n");
  std::vector<float>h_output2(n * batch, 0);
  cpu_layer_norm(
    h_input.data(), h_output2.data(), epsilon, n, batch,
    h_gamma.data(), h_beta.data()
  );

  // for (int b = 0; b < batch; ++b){
  //   for (int i = 0; i < n; ++i) {
  //     printf("res %f\t", h_output2[i + b * n]);
  //   }
  //   printf("\n");
  // }
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_gamma);
  cudaFree(d_beta);
  bool res = check_result(h_output.data(), h_output2.data(), n * batch);
  if (res) {
    printf(YELLOW"result is true"NONE);
  } else {
    printf(RED"result is false"NONE);
  }
}
*/