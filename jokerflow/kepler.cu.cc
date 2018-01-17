#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "kepler.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

// Define the CUDA kernel.
template <typename T>
__global__ void KeplerCudaKernel(const int size, const T* M, const T* e, T* E) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    E[i] = kepler<T>(ldg(M + i), ldg(e + i));
  }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
void KeplerFunctor<GPUDevice, T>::operator()(
    const GPUDevice& d, int size, const T* M, const T* e, T* E) {
  // Launch the cuda kernel.
  //
  // See core/util/cuda_kernel_helper.h for example of computing
  // block count and thread_per_block count.
  int block_count = 1024;
  int thread_per_block = 20;
  KeplerCudaKernel<T>
      <<<block_count, thread_per_block, 0, d.stream()>>>(size, M, e, E);
}

// Explicitly instantiate functors for the types of OpKernels registered.
template struct KeplerFunctor<GPUDevice, float>;

#endif  // GOOGLE_CUDA
