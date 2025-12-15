/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#include <stddef.h>
#include <cutlass/numeric_types.h>
#include <cuda_runtime_api.h>

#include "flash.h"
#include "exception.h"
#include "static_switch.h"
#include "check.h"

#include "flash_common.h"
#include "mha_fwd.h"

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel=false) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            if (params.num_splits <= 1 && !force_split_kernel) {  // If we don't set it num_splits == 0
                run_mha_fwd_<elem_type, kHeadDim>(params, stream);
            } else {
                run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim>(params, stream);
            }
        });
    });
}


void mha_fwd(cudaStream_t stream, void **buffers, const char* opaque, size_t opaque_len) {
	// buffers:
	//   &q,         // batch_size x seqlen_q x num_heads x head_size
	//   &k,         // batch_size x seqlen_k x num_heads_k x head_size
	//   &v,         // batch_size x seqlen_k x num_heads_k x head_size
	//   &out_,             // [batch_size,   seqlen_q,  num_heads,  head_size]
	//   &softmax_lse       // [batch_size,  num_heads,  seqlen_q]
	//  x &alibi_slopes_, // num_heads or batch_size x num_heads
	void* q = buffers[0];
	void* k = buffers[1];
	void* v = buffers[2];
	void* o = buffers[3];
	void* lse = buffers[4];
    void* cu_seqlens_q = buffers[5];
    void* cu_seqlens_k = buffers[6];

	mha_fwd_args args = Unpack<mha_fwd_args>(opaque, opaque_len);

	int device, major, minor;
	C10_CUDA_CHECK(cudaGetDevice(&device));
	C10_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
	C10_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));

    // bool is_sm75 = dprops->major == 7 && dprops->minor == 5;
    bool is_sm8x = major == 8 && minor >= 0;
    bool is_sm90 = major == 9 && minor == 0;
    FLASH_CHECK(is_sm90 || is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");
    // We will support Turing in the near future

    FLASH_CHECK(args.dtype == FP16 || args.dtype == BF16, "FlashAttention only support fp16 and bf16 data type");
    if (args.dtype == BF16) {
        FLASH_CHECK(is_sm90 || is_sm8x, "bfloat16 is only supported on Ampere GPUs or newer");
    }

    const int batch_size = args.n;
    int seqlen_q = args.l;
    int num_heads = args.h;
    const int head_size_og = args.d;
    const int seqlen_k = args.l_k;
    const int num_heads_k = args.h_k;
    FLASH_CHECK(batch_size > 0, "batch size must be postive");
    FLASH_CHECK(head_size_og <= 256, "FlashAttention forward only supports head dimension at most 256");
    FLASH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    if (args.window_size_left >= seqlen_k) { args.window_size_left = -1; }
    if (args.window_size_right >= seqlen_k) { args.window_size_right = -1; }

	bool has_alibi = false;

    // causal=true is the same as causal=false in this case
    if (seqlen_q == 1 && !has_alibi) { args.is_causal = false; }
	if (seqlen_q == 1) { args.is_causal = false; }
    if (args.is_causal) { args.window_size_right = 0; }

    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case

	void *q_padded=q, *k_padded=k, *v_padded=v;

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, 8);
    const int head_size_rounded = round_multiple(head_size, 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

	void* p = nullptr;
    if (args.return_softmax) {
		FLASH_CHECK(false, "no return softmax");
    }

    Flash_fwd_params params;
    set_params_fprop(params, args.dtype,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q_padded, k_padded, v_padded, o,
                     cu_seqlens_q,
                     cu_seqlens_k,
                     /*seqused_k=*/nullptr,
                     args.return_softmax ? p : nullptr,
                     lse,
                     args.p_dropout,
                     args.softmax_scale,
                     args.window_size_left,
                     args.window_size_right);


	int sm_count;
	C10_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));

    set_params_splitkv(params, batch_size, num_heads,
                       head_size, seqlen_k, seqlen_q,
                       head_size_rounded, args.p_dropout, /*num_splits*/0, sm_count, args.dtype);

    // number of times random will be generated per thread, to offset philox counter in thc random
	C10_CUDA_CHECK(cudaMalloc((void**)&params.rng_state, 2 * 8)); // 2 * float64

    if (args.p_dropout > 0.0)  {
		FLASH_CHECK(false, "don't support dropout yet");
    }


    if (has_alibi) {
    } else {
        params.alibi_slopes_ptr = nullptr;
    }

    if (seqlen_k > 0) {
		run_mha_fwd(params, stream);
    } else {
		FLASH_CHECK(false, "seqlen_k is zero");
    }

	C10_CUDA_CHECK(cudaFree(params.rng_state));
	if(params.softmax_lseaccum_ptr != nullptr) {
		C10_CUDA_CHECK(cudaFree(params.softmax_lseaccum_ptr));
		C10_CUDA_CHECK(cudaFree(params.oaccum_ptr));
	}
}

