#include <stddef.h>
#include <cutlass/numeric_types.h>
#include <cuda_runtime_api.h>
#include <cute/layout.hpp>
#include <cstdio>

#include "flash.h"
#include "exception.h"
#include "static_switch.h"
#include "check.h"

#include "flash_common.h"
#include "mha_bwd.h"

void set_params_dgrad(Flash_bwd_params &params,
					  ElementType element_type,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      void* q_ptr,
                      void* k_ptr,
                      void* v_ptr,
                      void* out_ptr,
                      void* dout_ptr,
                      void *dq_ptr,
                      void *dk_ptr,
                      void *dv_ptr,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *dq_accum_d,
                      void *dk_accum_d,
                      void *dv_accum_d,
                      void *softmax_lse_d,
                      void *dsoftmax_sum_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      bool deterministic) {

    set_params_fprop(params, element_type,
                     b, seqlen_q, seqlen_k, seqlen_q_rounded, seqlen_k_rounded, h, h_k, d, d_rounded,
                     q_ptr, k_ptr, v_ptr, out_ptr,
                     cu_seqlens_q_d,
                     cu_seqlens_k_d,
                     nullptr,
                     nullptr,
                     softmax_lse_d,
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right);

    // Set the pointers and strides.
    params.do_ptr = dout_ptr;
	params.do_row_stride = params.o_row_stride;
	params.do_head_stride = params.o_head_stride;
    params.dq_ptr = dq_ptr;
    params.dk_ptr = dk_ptr;
    params.dv_ptr = dv_ptr;

    // dk&dv is expanded to the same h as dq for MQA, we sum it later
    auto dq = cute::compact_row_major(cute::make_shape(b, seqlen_q, h, d));
	auto dk = cute::compact_row_major(cute::make_shape(b, seqlen_k, h, d));
	auto dv = cute::compact_row_major(cute::make_shape(b, seqlen_k, h, d));

    params.dq_row_stride = cute::get<1>(dq);
    params.dk_row_stride = cute::get<1>(dk);
    params.dv_row_stride = cute::get<1>(dv);
    params.dq_head_stride = cute::get<2>(dq);
    params.dk_head_stride = cute::get<2>(dk);
    params.dv_head_stride = cute::get<2>(dv);

    if (cu_seqlens_q_d == nullptr) {
        params.do_batch_stride = params.o_batch_stride;
        params.dq_batch_stride = cute::get<0>(dq);
        params.dk_batch_stride = cute::get<0>(dk);
        params.dv_batch_stride = cute::get<0>(dv);
    }

    params.dq_accum_ptr = dq_accum_d;
    params.dk_accum_ptr = dk_accum_d;
    params.dv_accum_ptr = dv_accum_d;

    // Softmax sum
    params.dsoftmax_sum = dsoftmax_sum_d;

    params.deterministic = deterministic;
}

void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            run_mha_bwd_<elem_type, kHeadDim>(params, stream);
        });
    });
}

void
mha_bwd(cudaStream_t stream, void **buffers, const char* opaque, size_t opaque_len) {
	// (const at::Tensor &dout,  // batch_size x seqlen_q x num_heads, x head_size_og
    //     const at::Tensor &q,   // batch_size x seqlen_q x num_heads x head_size
    //     const at::Tensor &k,   // batch_size x seqlen_k x num_heads_k x head_size
    //     const at::Tensor &v,   // batch_size x seqlen_k x num_heads_k x head_size
    //     const at::Tensor &out,   // batch_size x seqlen_q x num_heads x head_size
    //     const at::Tensor &softmax_lse,     // b x h x seqlen_q
    //     c10::optional<at::Tensor> &dq_,   // batch_size x seqlen_q x num_heads x head_size
    //     c10::optional<at::Tensor> &dk_,   // batch_size x seqlen_k x num_heads_k x head_size
    //     c10::optional<at::Tensor> &dv_,   // batch_size x seqlen_k x num_heads_k x head_size
    //     c10::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
	void* dout = buffers[0];
	void* q = buffers[1];
	void* k = buffers[2];
	void* v = buffers[3];
	void* o = buffers[4];
	void* lse = buffers[5];

	void* dq = buffers[6];
	void* dk = buffers[7];
	void* dv = buffers[8];
    void* cu_seqlens_q = buffers[9];
    void* cu_seqlens_k = buffers[10];

	auto args = Unpack<mha_bwd_args>(opaque, opaque_len);

	int window_size_right = args.window_size_right;
	int window_size_left = args.window_size_left;

	int device, major, minor, sm_count;
	C10_CUDA_CHECK(cudaGetDevice(&device));
	C10_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
	C10_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
	C10_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));


    if (args.is_causal) { window_size_right = 0; }
    // auto dprops = at::cuda::getCurrentDeviceProperties();
    // bool is_sm75 = dprops->major == 7 && dprops->minor == 5;
    bool is_sm8x = major == 8 && minor >= 0;
    bool is_sm80 = major == 8 && minor == 0;
    bool is_sm90 = major == 9 && minor == 0;
    FLASH_CHECK(is_sm90 || is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");
    // We will support Turing in the near future

    bool is_dropout = args.p_dropout > 0.0;

    auto q_dtype = args.dtype;

    if (q_dtype == BF16) {
        FLASH_CHECK(is_sm90 || is_sm8x, "bfloat16 is only supported on Ampere GPUs or newer");
    }

    const int batch_size = args.n;
    const int seqlen_q = args.l;
    const int num_heads = args.h;
    const int head_size_og = args.d; //dout.size(3);
    const int head_size = args.d + (8 - head_size_og%8) % 8; //sizes[3];
    const int seqlen_k = args.l_k;
    const int num_heads_k = args.h_k; //k.size(2);
    FLASH_CHECK(batch_size > 0, "batch size must be positive");
    FLASH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
    FLASH_CHECK(head_size <= 256, "FlashAttention backward only supports head dimension at most 256");
    if (head_size > 192) {
        FLASH_CHECK(is_sm80 || is_sm90, "FlashAttention backward for head dim > 192 requires A100/A800 or H100/H800");
    }
    FLASH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    FLASH_CHECK(head_size == round_multiple(head_size_og, 8), "head_size must be head_size_og rounded to a multiple of 8");

    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }

    // bool loop = seqlen_k > blocksize_c;
    // TODO: change later, for now set to true for simplicity
    bool loop = true;

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing

    void* softmax_d = nullptr;
	C10_CUDA_CHECK(cudaMalloc(&softmax_d, batch_size * num_heads * seqlen_q_rounded * 4));
    void* dq_accum = nullptr;
    void* dk_accum = nullptr;
	void* dv_accum = nullptr;
    if (loop) {
        // IMPORTANT:
        // When `cu_seqlens_q` is non-null, the backward CUDA kernels index `dq_accum` using
        // an additional `+ 128 * bidb` term (see `flash_bwd_preprocess_kernel.h` and
        // `flash_bwd_kernel.h`). This requires allocating `dq_accum` with extra space.
        //
        // Allocate rows:
        //   dq_accum_rows = total_q + 128 * batch_size
        //
        // For fixed-length sequences, total_q = batch_size * seqlen_q. This is also an upper
        // bound for variable-length if seqlen_q is the max length.
        const bool has_cu_seqlens = (cu_seqlens_q != nullptr) && (cu_seqlens_k != nullptr);
        const size_t total_q = static_cast<size_t>(batch_size) * static_cast<size_t>(seqlen_q);
        const size_t dq_accum_rows = has_cu_seqlens
            ? (total_q + 128ull * static_cast<size_t>(batch_size))
            : (static_cast<size_t>(batch_size) * static_cast<size_t>(seqlen_q_rounded));
        const size_t dq_accum_elems_per_row = static_cast<size_t>(num_heads) * static_cast<size_t>(head_size_rounded);
        const size_t dq_accum_bytes_per_split = dq_accum_rows * dq_accum_elems_per_row * sizeof(float);

        const char* dbg = std::getenv("FLASH_ATTN_TF_DEBUG_BWD");
        if (dbg && dbg[0] == '1') {
            std::fprintf(
                stderr,
                "[flash_attn_tf][mha_bwd] batch=%d seqlen_q=%d seqlen_q_rounded=%d heads=%d head_size_rounded=%d "
                "has_cu_seqlens=%d total_q=%zu dq_accum_rows=%zu dq_accum_bytes_per_split=%zu\\n",
                batch_size, seqlen_q, seqlen_q_rounded, num_heads, head_size_rounded,
                int(has_cu_seqlens), total_q, dq_accum_rows, dq_accum_bytes_per_split
            );
        }

        if (!args.deterministic) {
			C10_CUDA_CHECK(cudaMalloc(&dq_accum, dq_accum_bytes_per_split));
			C10_CUDA_CHECK(cudaMemset(dq_accum, 0, dq_accum_bytes_per_split));
        } else {
            const int nsplits = (sm_count + batch_size * num_heads - 1) / (batch_size * num_heads);
			C10_CUDA_CHECK(cudaMalloc(&dq_accum, static_cast<size_t>(nsplits) * dq_accum_bytes_per_split));
			// previously allocated with torch.zeros, so i guess we need to zero it
			C10_CUDA_CHECK(cudaMemset(dq_accum, 0, static_cast<size_t>(nsplits) * dq_accum_bytes_per_split));
        }
    }


    // For MQA, dk and dv are expanded to the same n_heads as dq (handled in xla).
    // After returning the result, it gets reduced to the original size by summing, so we don't need to do anything here.
	void* dk_expanded = dk;
	void* dv_expanded = dv;

    Flash_bwd_params params;

    set_params_dgrad(params,
					 args.dtype,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q, k, v, o,
                     dout, dq, dk_expanded, dv_expanded,
                     cu_seqlens_q,
                     cu_seqlens_k,
                     loop ? dq_accum : nullptr,
                     nullptr,
                     nullptr,
                     lse,
                     softmax_d,
                     args.p_dropout,
                     args.softmax_scale,
                     window_size_left,
                     window_size_right,
                     args.deterministic);
    // The CUDA kernels treat `dq_accum_split_stride` as an element stride between split buffers.
    // It must match the number of elements in one dq_accum buffer (not bytes).
    if (!args.deterministic) {
        params.dq_accum_split_stride = 0;
    } else {
        const bool has_cu_seqlens = (cu_seqlens_q != nullptr) && (cu_seqlens_k != nullptr);
        const size_t total_q = static_cast<size_t>(batch_size) * static_cast<size_t>(seqlen_q);
        const size_t dq_accum_rows = has_cu_seqlens
            ? (total_q + 128ull * static_cast<size_t>(batch_size))
            : (static_cast<size_t>(batch_size) * static_cast<size_t>(seqlen_q_rounded));
        const size_t dq_accum_elems_per_row = static_cast<size_t>(num_heads) * static_cast<size_t>(head_size_rounded);
        params.dq_accum_split_stride = static_cast<int64_t>(dq_accum_rows * dq_accum_elems_per_row);
    }

    auto launch = &run_mha_bwd;

	C10_CUDA_CHECK(cudaMalloc((void**)&params.rng_state, 2 * 8)); // 2 * float64
    // if (is_dropout)  {
	// 	FLASH_CHECK(false, "don't support dropout yet");
    // }



    if (seqlen_q > 0) {
        launch(params, stream);
    } else {
		FLASH_CHECK(false, "seqlen_q == 0");
    }

	C10_CUDA_CHECK(cudaFree(params.rng_state));
	if(softmax_d != nullptr) {
		C10_CUDA_CHECK(cudaFree(softmax_d));
	}
	if(dq_accum != nullptr) {
		C10_CUDA_CHECK(cudaFree(dq_accum));
	}
	if(dk_accum != nullptr) {
		C10_CUDA_CHECK(cudaFree(dk_accum));
	}
	if(dv_accum != nullptr) {
		C10_CUDA_CHECK(cudaFree(dv_accum));
	}

    // For MQA/GQA we need to sum dK and dV across the groups
    if (num_heads_k != num_heads) {
		// FLASH_CHECK(false, "don't handle MQA yet");
    }
}

