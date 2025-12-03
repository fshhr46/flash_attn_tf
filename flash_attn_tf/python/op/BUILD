package(
    default_visibility = [
        "//flash_attn_tf:__subpackages__",
    ],
)

py_library(
    name = "flash_attn_ops",
    srcs = glob(["*.py"]),
    data = [
        "//flash_attn_tf/core:_flash_attn_ops.so",
    ],
    srcs_version = "PY3",
)
