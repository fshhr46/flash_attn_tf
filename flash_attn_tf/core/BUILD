load("//flash_attn_tf:flash_attn_tf.bzl", "custom_op_library")

package(default_visibility = ["//visibility:public"])

custom_op_library(
    name = "_flash_attn_ops.so",
    srcs = [
        "flash_attn/flash_api.cpp",
        "flash_attn/mha_fwd.cpp",
        "flash_attn/mha_bwd.cpp",
        "flash_attn/flash_common.cpp",
        "flash_attn/check.h",
        "flash_attn/flash_common.h",
        "flash_attn/mha_fwd.h",
        "flash_attn/mha_bwd.h",
    ],
    deps = [
        "//flash_attn_tf/core/flash_attn/src:flash_attn_kernels",
        "@cutlass//:cutlass",
    ],
)
