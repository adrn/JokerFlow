#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <cmath>

using namespace tensorflow;

REGISTER_OP("KeplerGrad")
  .Attr("T: {float, double}")
  .Input("manom: T")
  .Input("eccen: T")
  .Input("eanom: T")
  .Input("beanom: T")
  .Output("bmanom: T")
  .Output("beccen: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {

    ::tensorflow::shape_inference::ShapeHandle M, e, E, bE;

    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &M));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &e));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &E));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &bE));
    TF_RETURN_IF_ERROR(c->Merge(M, e, &M));
    TF_RETURN_IF_ERROR(c->Merge(M, E, &M));
    TF_RETURN_IF_ERROR(c->Merge(M, bE, &M));

    c->set_output(0, c->input(0));
    c->set_output(1, c->input(1));

    return Status::OK();
  });

template <typename T>
class KeplerGradOp : public OpKernel {
 public:
  explicit KeplerGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Inputs
    const Tensor& M_tensor = context->input(0);
    const Tensor& e_tensor = context->input(1);
    const Tensor& E_tensor = context->input(2);
    const Tensor& bE_tensor = context->input(3);

    // Dimensions
    const int64 N = M_tensor.dim_size(0);
    OP_REQUIRES(context, e_tensor.dim_size(0) == N, errors::InvalidArgument("dimension mismatch"));
    OP_REQUIRES(context, E_tensor.dim_size(0) == N, errors::InvalidArgument("dimension mismatch"));
    OP_REQUIRES(context, bE_tensor.dim_size(0) == N, errors::InvalidArgument("dimension mismatch"));
    OP_REQUIRES(context, N <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));

    // Output
    Tensor* bM_tensor = NULL, * be_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({N}), &bM_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({N}), &be_tensor));

    // Access the data
    const auto e = e_tensor.template flat<T>();
    const auto E = E_tensor.template flat<T>();
    const auto bE = bE_tensor.template flat<T>();
    auto bM = bM_tensor->template flat<T>();
    auto be = be_tensor->template flat<T>();

    for (int64 n = 0; n < N; ++n) {
      bM(n) = bE(n) / (T(1.0) - e(n) * cos(E(n)));
      be(n) = sin(E(n)) * bM(n);
    }
  }
};

#define REGISTER_KERNEL(type)                                              \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("KeplerGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      KeplerGradOp<type>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL
