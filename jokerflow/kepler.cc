#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/work_sharder.h"

#include <cmath>

#include "kepler.h"

using namespace tensorflow;
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("Kepler")
  .Attr("T: {float, double}")
  .Input("manom: T")
  .Input("eccen: T")
  .Output("eanom: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {

    ::tensorflow::shape_inference::ShapeHandle M, e;

    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &M));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &e));
    TF_RETURN_IF_ERROR(c->Merge(M, e, &M));
    c->set_output(0, c->input(0));

    return Status::OK();
  });

// Implementation of forward Kepler function

template <typename T>
struct KeplerFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, int size, const T* M, const T* e, T* E) {
    for (int n = 0; n < size; ++n) {
      E[n] = kepler<T>(M[n], e[n]);
    }
  }
};

template <typename Device, typename T>
class KeplerOp : public OpKernel {
 public:
  explicit KeplerOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Inputs
    const Tensor& M_tensor = context->input(0);
    const Tensor& e_tensor = context->input(1);

    // Dimensions
    const int64 N = M_tensor.dim_size(0);
    OP_REQUIRES(context, e_tensor.dim_size(0) == N, errors::InvalidArgument("dimension mismatch"));
    OP_REQUIRES(context, N <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));

    // Output
    Tensor* E_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({N}), &E_tensor));

    // Access the data
    const auto M = M_tensor.template flat<T>();
    const auto e = e_tensor.template flat<T>();
    auto E = E_tensor->template flat<T>();

    KeplerFunctor<Device, T>()(
        context->eigen_device<Device>(),
        static_cast<int>(N), M.data(), e.data(), E.data());
  }
};


#define REGISTER_KERNEL(type)                                              \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("Kepler").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      KeplerOp<CPUDevice, type>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL

#ifdef GOOGLE_CUDA

extern template KeplerFunctor<GPUDevice, float>;
REGISTER_KERNEL_BUILDER(
    Name("Kepler").Device(DEVICE_GPU).TypeConstraint<T>("T"),
    ExampleOp<GPUDevice, T>);

#endif  // GOOGLE_CUDA
