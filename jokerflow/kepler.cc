#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/work_sharder.h"

#include <cmath>

#define KEPLER_MAX_ITER 200
#define KEPLER_TOL      1.234e-10

using namespace tensorflow;

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


template <typename T>
inline T kepler (const T& M, const T& e) {
  T E0 = M, E = M;
  for (int i = 0; i < KEPLER_MAX_ITER; ++i) {
    T g = E0 - e * sin(E0) - M, gp = 1.0 - e * cos(E0);
    E = E0 - g / gp;
    if (std::abs((E - E0) / E) <= T(KEPLER_TOL)) return E;
    E0 = E;
  }

  // If we get here, we didn't converge, but return the best estimate.
  return E;
}

template <typename T>
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

    // Output
    Tensor* E_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({N}), &E_tensor));

    // Access the data
    auto M = M_tensor.template flat<T>();
    auto e = e_tensor.template flat<T>();
    auto E = E_tensor->template flat<T>();

    for (int64 n = 0; n < N; ++n) {
      E(n) = kepler<T>(M(n), e(n));
    }

    // Could maybe parallelize on the CPU...
    //auto pool = context->device()->tensorflow_cpu_worker_threads()->workers;
    //Shard(pool->NumThreads(), pool, N, 10, [&](int64 start, int64 end) {
    //  for(int64 n = start; n < end; ++n) {
    //    E(n) = kepler<T>(M(n), e(n));
    //  }
    //});

  }
};


#define REGISTER_KERNEL(type)                                              \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("Kepler").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      KeplerOp<type>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL
#undef KEPLER_MAX_ITER
#undef KEPLER_TOL
