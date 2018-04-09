#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "kepler_op.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

// Define the CUDA kernel.
template <typename T>
__global__ void KeplerCudaKernel(const int maxiter, const float tol, const int size, const T* M, const T* e, T* E) {
  int n;
  T e_, M_, E0, E_, s, c, g, gp;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
    e_ = ldg(e + i);
    M_ = ldg(M + i);
    if (fabsf(e_) < tol) {
      E[i] = M_;
    } else {
      E0 = M_;
      E_ = E0;
      for (n = 0; n < maxiter; ++n) {
        sincosf(E0, &s, &c);
        g = E0 - e_ * s - M_;
        gp = 1.0 - e_ * c;
        E_ = E0 - g / gp;
        if (fabsf((E_ - E0) / E_) <= tol) {
          E[i] = E_;
          n = maxiter;
        }
        E0 = E_;
      }
    }
  }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
void KeplerFunctor<GPUDevice, T>::operator()(
    const GPUDevice& d, int maxiter, float tol, int size, const T* M, const T* e, T* E) {
  // Launch the cuda kernel.
  //
  // See core/util/cuda_kernel_helper.h for example of computing
  // block count and thread_per_block count.
  int block_count = 1024;
  int thread_per_block = 20;
  KeplerCudaKernel<T>
      <<<block_count, thread_per_block, 0, d.stream()>>>(maxiter, tol, size, M, e, E);
}

// Explicitly instantiate functors for the types of OpKernels registered.
template struct KeplerFunctor<GPUDevice, float>;

#endif  // GOOGLE_CUDA
