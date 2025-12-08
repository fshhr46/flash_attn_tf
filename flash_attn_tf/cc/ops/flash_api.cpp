#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/stream_executor.h"

#include "mha_fwd.h"
#include "flash_common.h"

using namespace tensorflow;

REGISTER_OP("FlashMHAFwd")
    .Input("q: T")
    .Input("k: T")
    .Input("v: T")
    .Input("softmax_scale: float32")
    .Input("is_causal: bool")
    .Input("window_size_left: int32")
    .Input("window_size_right: int32")
    .Input("return_softmax: bool")
    .Input("n: int32")
    .Input("l: int32")
    .Input("h: int32")
    .Input("d: int32")
    .Input("l_k: int32")
    .Input("h_k: int32")
    .Input("dtype: int32")
    .Input("seed: int64")
    .Output("output: T")
    .Output("softmax_lse: float32")
    .Attr("T: {half, bfloat16}") // BF16 or FP16
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        // q: [batch_size, seqlen_q, nheads, headdim]
        shape_inference::ShapeHandle q_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &q_shape));

        // Output has same shape as q
        c->set_output(0, q_shape);

        // softmax_lse: [batch_size, nheads, seqlen_q]
        shape_inference::DimensionHandle batch_size = c->Dim(q_shape, 0);
        shape_inference::DimensionHandle seqlen_q = c->Dim(q_shape, 1);
        shape_inference::DimensionHandle nheads = c->Dim(q_shape, 2);

        c->set_output(1, c->MakeShape({batch_size, nheads, seqlen_q}));

        return absl::OkStatus();
    })
    .Doc(R"doc(
FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.
)doc");

template <typename T>
class FlashMHAFwdOp : public OpKernel {
 public:
  explicit FlashMHAFwdOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Inputs
    const Tensor& q = context->input(0);
    const Tensor& k = context->input(1);
    const Tensor& v = context->input(2);

    float softmax_scale = context->input(3).scalar<float>()();
    bool is_causal = context->input(4).scalar<bool>()();
    int window_size_left = context->input(5).scalar<int>()();
    int window_size_right = context->input(6).scalar<int>()();
    bool return_softmax = context->input(7).scalar<bool>()();
    int n = context->input(8).scalar<int>()();
    int l = context->input(9).scalar<int>()();
    int h = context->input(10).scalar<int>()();
    int d = context->input(11).scalar<int>()();
    int l_k = context->input(12).scalar<int>()();
    int h_k = context->input(13).scalar<int>()();
    // input(14) is dtype, which we ignore because we rely on template T
    uint64_t seed = context->input(15).scalar<int64>()();

    // Outputs
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, q.shape(), &output));
    
    Tensor* softmax_lse = nullptr;
    // LSE shape: [batch_size, nheads, seqlen_q]
    TensorShape lse_shape({n, h, l}); 
    OP_REQUIRES_OK(context, context->allocate_output(1, lse_shape, &softmax_lse));

    // Construct args
    mha_fwd_args args;
    args.p_dropout = 0.0f; 
    args.softmax_scale = softmax_scale;
    args.is_causal = is_causal;
    args.window_size_left = window_size_left;
    args.window_size_right = window_size_right;
    args.return_softmax = return_softmax;
    args.n = n;
    args.l = l;
    args.h = h;
    args.d = d;
    args.l_k = l_k;
    args.h_k = h_k;
    
    if (std::is_same<T, Eigen::half>::value) {
        args.dtype = ElementType::FP16;
    } else {
        args.dtype = ElementType::BF16;
    }
    args.seed = seed;

    void* buffers[] = {
        (void*)q.data(),
        (void*)k.data(),
        (void*)v.data(),
        (void*)output->data(),
        (void*)softmax_lse->data()
    };

    std::string opaque = Pack(args);
    
    // Get CUDA stream
    auto stream_ptr = context->op_device_context()->stream();
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr->platform_specific_handle().stream);
    
    mha_fwd(stream, buffers, opaque.c_str(), opaque.size());
  }
};

REGISTER_KERNEL_BUILDER(
    Name("FlashMHAFwd").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"),
    FlashMHAFwdOp<Eigen::half>);

REGISTER_KERNEL_BUILDER(
    Name("FlashMHAFwd").Device(DEVICE_GPU).TypeConstraint<bfloat16>("T"),
    FlashMHAFwdOp<bfloat16>);
