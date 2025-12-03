load("//flash_attn_tf:flash_attn_tf.bzl", "custom_op_library")

package(default_visibility = ["//visibility:public"])

custom_op_library(
    name = "_flash_attn_ops.so",
    srcs = [
        "flash_attn/flash_api.cpp",
    ],
    deps = [
        "@local_config_tf//:tf_header_lib",
    ],
)
