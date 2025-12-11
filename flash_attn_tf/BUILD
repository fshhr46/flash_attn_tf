licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda_is_configured", "if_cuda")

cc_library(
    name = "cuda",
    data = [
        "@local_config_cuda//cuda:cudart",
    ],
    linkopts = select({
        "@local_config_cuda//cuda:darwin": [
            "-Wl,-rpath,../local_config_cuda/cuda/lib",
            "-Wl,-rpath,../local_config_cuda/cuda/extras/CUPTI/lib",
        ],
        "//conditions:default": [
            "-Wl,-rpath,../local_config_cuda/cuda/lib64",
            "-Wl,-rpath,../local_config_cuda/cuda/extras/CUPTI/lib64",
        ],
    }),
    deps = [
        "@local_config_cuda//cuda:cudart",
    ],
)

cc_library(
    name = "flash_attn_ops_gpu",
    srcs = [
        "cc/kernels/flash_ops.cc",
        "cc/kernels/flash_bwd_hdim32_bf16_sm80.cu.cc",
        "cc/kernels/flash_fwd_hdim32_bf16_sm80.cu.cc",
        "cc/kernels/flash_fwd_split_hdim32_bf16_sm80.cu.cc",
    ] + glob([
        # "cc/kernels/flash_*.cu.cc",
        "cc/ops/*.cpp",
    ]),
    hdrs = glob([
        "cc/kernels/*.h",
        "cc/kernels/*.cuh",
        "cc/ops/*.h",
    ]),
    includes = ["cc/kernels", "cc/ops"],
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ] + if_cuda_is_configured([
        ":cuda",
        "@local_config_cuda//cuda:cuda_headers",
        "@cutlass//:cutlass"
    ]),
    alwayslink = 1,
    copts = select({
        "//conditions:default": ["-pthread", "-std=c++17", "-D_GLIBCXX_USE_CXX11_ABI=1"],
    }) + if_cuda_is_configured(["-DTENSORFLOW_USE_NVCC=1", "-DGOOGLE_CUDA=1", "-x cuda", "-nvcc_options=relaxed-constexpr", "-nvcc_options=ftz=true"]),
)


cc_binary(
    name = "python/ops/_flash_attn_ops.so",
    srcs = [
        "cc/kernels/flash_ops.cc",
        "cc/ops/flash_common.h",
        "cc/ops/mha_fwd.h",
        "cc/ops/mha_bwd.h",
    ],
    linkshared = 1,
    deps = [
        ":flash_attn_ops_gpu",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ] + if_cuda_is_configured([":cuda"]),
    copts = select({
        "//conditions:default": ["-pthread", "-std=c++17", "-D_GLIBCXX_USE_CXX11_ABI=1"],
    }) + if_cuda_is_configured(["-DTENSORFLOW_USE_NVCC=1", "-DGOOGLE_CUDA=1"]),
)

py_library(
    name = "flash_attn_ops_py",
    srcs = ([
        "python/ops/flash_attn_ops.py",
    ]),
    data = [
        ":python/ops/_flash_attn_ops.so"
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "flash_attn_ops_test",
    srcs = [
        "python/ops/flash_attn_ops_test.py"
    ],
    main = "python/ops/flash_attn_ops_test.py",
    deps = [
        ":flash_attn_ops_py",
    ],
    srcs_version = "PY2AND3",
)

py_library(
    name = "flash_attn_tf_py",
    srcs = ([
        "__init__.py",
        "python/__init__.py",
        "python/ops/__init__.py",
    ]),
    deps = [
        ":flash_attn_ops_py",
    ],
    srcs_version = "PY2AND3",
)
