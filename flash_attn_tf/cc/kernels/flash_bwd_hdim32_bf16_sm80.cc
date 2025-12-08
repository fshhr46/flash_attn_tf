#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/stream_executor.h"

#include "mha_bwd.h"
#include "flash_common.h"

using namespace tensorflow;

REGISTER_OP("FlashMHABwd")
    .Input("dout: T")
    .Input("q: T")
    .Input("k: T")
    .Input("v: T")
    .Input("out: T")
    .Input("softmax_lse: float32")
    .Input("p_dropout: float32")
    .Input("softmax_scale: float32")
    .Input("is_causal: bool")
    .Input("window_size_left: int32")
    .Input("window_size_right: int32")
    .Input("deterministic: bool")
    .Input("n: int32")
    .Input("l: int32")
    .Input("h: int32")
    .Input("d: int32")
    .Input("l_k: int32")
    .Input("h_k: int32")
    .Input("seed: int64")
    .Output("dq: T")
    .Output("dk: T")
    .Output("dv: T")
    .Attr("T: {half, bfloat16}")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        // dout: [batch_size, seqlen_q, nheads, headdim]
        shape_inference::ShapeHandle dout_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &dout_shape));
        
        // q: [batch_size, seqlen_q, nheads, headdim]
        shape_inference::ShapeHandle q_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &q_shape));

        // k: [batch_size, seqlen_k, nheads_k, headdim]
        shape_inference::ShapeHandle k_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 4, &k_shape));
        
        // v: [batch_size, seqlen_k, nheads_k, headdim]
        shape_inference::ShapeHandle v_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 4, &v_shape));

        // dq same shape as q
        c->set_output(0, q_shape);
        // dk same shape as k
        c->set_output(1, k_shape);
        // dv same shape as v (which is same as k usually)
        c->set_output(2, v_shape);

        return absl::OkStatus();
    });

template <typename T>
class FlashMHABwdOp : public OpKernel {
 public:
  explicit FlashMHABwdOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Inputs
    const Tensor& dout = context->input(0);
    const Tensor& q = context->input(1);
    const Tensor& k = context->input(2);
    const Tensor& v = context->input(3);
    const Tensor& out = context->input(4);
    const Tensor& softmax_lse = context->input(5);

    float p_dropout = context->input(6).scalar<float>()();
    float softmax_scale = context->input(7).scalar<float>()();
    bool is_causal = context->input(8).scalar<bool>()();
    int window_size_left = context->input(9).scalar<int>()();
    int window_size_right = context->input(10).scalar<int>()();
    bool deterministic = context->input(11).scalar<bool>()();
    int n = context->input(12).scalar<int>()();
    int l = context->input(13).scalar<int>()();
    int h = context->input(14).scalar<int>()();
    int d = context->input(15).scalar<int>()();
    int l_k = context->input(16).scalar<int>()();
    int h_k = context->input(17).scalar<int>()();
    uint64_t seed = context->input(18).scalar<int64>()();

    // Outputs
    Tensor* dq = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, q.shape(), &dq));
    
    Tensor* dk = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, k.shape(), &dk));

    Tensor* dv = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(2, v.shape(), &dv));

    // Construct args
    mha_bwd_args args;
    args.p_dropout = p_dropout;
    args.softmax_scale = softmax_scale;
    args.is_causal = is_causal;
    args.window_size_left = window_size_left;
    args.window_size_right = window_size_right;
    args.deterministic = deterministic;
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
        (void*)dout.data(),
        (void*)q.data(),
        (void*)k.data(),
        (void*)v.data(),
        (void*)out.data(),
        (void*)softmax_lse.data(),
        (void*)dq->data(),
        (void*)dk->data(),
        (void*)dv->data()
    };

    std::string opaque = Pack(args);
    
    // Get CUDA stream
    auto stream_ptr = context->op_device_context()->stream();
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr->platform_specific_handle().stream);
    
    mha_bwd(stream, buffers, opaque.c_str(), opaque.size());
  }
};

REGISTER_KERNEL_BUILDER(
    Name("FlashMHABwd").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"),
    FlashMHABwdOp<Eigen::half>);

REGISTER_KERNEL_BUILDER(
    Name("FlashMHABwd").Device(DEVICE_GPU).TypeConstraint<bfloat16>("T"),
    FlashMHABwdOp<bfloat16>);

