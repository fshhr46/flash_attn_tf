sh_binary(
    name = "build_pip_pkg",
    srcs = ["build_deps/build_pip_pkg.sh"],
    data = [
        "MANIFEST.in",
        "requirements.txt",
        "setup.py",
        "//flash_attn_tf:time_two_py",
    ],
)

filegroup(
    name = "clang_tidy_config",
    srcs = [".clang-tidy"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "clang_tidy_executable",
    srcs = ["@llvm_toolchain//:bin/clang-tidy"],
)
