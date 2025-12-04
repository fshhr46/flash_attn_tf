load("@rules_cuda//cuda:defs.bzl", "cuda_library")

package(default_visibility = ["//visibility:public"])

cuda_library(
    name = "flash_attn_kernels",
    srcs = glob(["*.cu"]),
    hdrs = glob(["*.h", "*.cuh"]),
    includes = ["."],
    deps = [
        "@cutlass//:cutlass",
        
    ],
)

#  /usr/local/cuda-12.4/bin/nvcc -I/home/hhuang2/flash_attn_tf/csrc/flash_attn -I/home/hhuang2/flash_attn_tf/csrc/flash_attn/src -I/home/hhuang2/flash_attn_tf/csrc/cutlass/include -I/home/hhuang2/anaconda3/envs/flat/lib/python3.10/site-packages/pybind11/include -I/usr/local/cuda-12.4/include -I/home/hhuang2/anaconda3/envs/flat/include/python3.10 -c csrc/flash_attn/src/flash_fwd_hdim32_bf16_sm80.cu -o build/temp.linux-x86_64-cpython-310/csrc/flash_attn/src/flash_fwd_hdim32_bf16_sm80.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options '-fPIC' -O3 -std=c++17 -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -gencode arch=compute_80,code=sm_80 --threads 32