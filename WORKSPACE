workspace(name = "flash_attn_tf")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# ==================
# protobuf toolchain
# ==================

git_repository(
    name = "com_google_protobuf",
    commit = "43e1626812c1b543e56a7bec59dc09eb18248bd2",  # v30.2
    remote = "https://github.com/protocolbuffers/protobuf",
)

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

load("@rules_java//java:rules_java_deps.bzl", "rules_java_dependencies")

rules_java_dependencies()

load("@rules_java//java:repositories.bzl", "rules_java_toolchains")

rules_java_toolchains()

load("@rules_python//python:repositories.bzl", "py_repositories")

py_repositories()

# ==============
# LLVM toolchain
# ==============

http_archive(
    name = "toolchains_llvm",
    canonical_id = "v1.4.0",
    sha256 = "fded02569617d24551a0ad09c0750dc53a3097237157b828a245681f0ae739f8",
    strip_prefix = "toolchains_llvm-v1.4.0",
    url = "https://github.com/bazel-contrib/toolchains_llvm/releases/download/v1.4.0/toolchains_llvm-v1.4.0.tar.gz",
)

load("@toolchains_llvm//toolchain:deps.bzl", "bazel_toolchain_dependencies")

bazel_toolchain_dependencies()

load("@toolchains_llvm//toolchain:rules.bzl", "llvm_toolchain")

llvm_toolchain(
    name = "llvm_toolchain",
    llvm_version = "15.0.6",
)

load("@llvm_toolchain//:toolchains.bzl", "llvm_register_toolchains")

llvm_register_toolchains()

# ============
# Hedron setup
# ============

# Hedron is a C++ code generation tool that generates compile_commands.json
# files for Bazel projects. It is used to improve IDE support for Bazel projects
# by providing accurate code completion, navigation, and refactoring capabilities.
# See https://github.com/hedronvision/bazel-compile-commands-extractor for details.
git_repository(
    name = "hedron_compile_commands",
    commit = "4f28899228fb3ad0126897876f147ca15026151e",
    remote = "https://github.com/hedronvision/bazel-compile-commands-extractor/",
)

load("@hedron_compile_commands//:workspace_setup.bzl", "hedron_compile_commands_setup")

hedron_compile_commands_setup()

load("@hedron_compile_commands//:workspace_setup_transitive.bzl", "hedron_compile_commands_setup_transitive")

hedron_compile_commands_setup_transitive()

load("@hedron_compile_commands//:workspace_setup_transitive_transitive.bzl", "hedron_compile_commands_setup_transitive_transitive")

hedron_compile_commands_setup_transitive_transitive()

load("@hedron_compile_commands//:workspace_setup_transitive_transitive_transitive.bzl", "hedron_compile_commands_setup_transitive_transitive_transitive")

hedron_compile_commands_setup_transitive_transitive_transitive()

# ================
# clang-tidy setup
# ================

git_repository(
    name = "bazel_clang_tidy",
    commit = "07bc38524e61d9501d772726d6e27cb980db42c7",
    remote = "https://github.com/erenon/bazel_clang_tidy",
)

# ================
# TensorFlow setup
# ================

load("//build_deps/tf_dependency:tf_configure.bzl", "tf_configure")

tf_configure(
    name = "local_config_tf",
)


# =================
# CUDA setup
# ================
load("//build_deps/gpu:cuda_configure.bzl", "cuda_configure")

cuda_configure(name = "local_config_cuda")


# ==================
# Abseil C++ library
# ==================
# set it to 20230802.3 to match Tensorflow 2.16.2
http_archive(
    name = "com_google_absl",
    sha256 = "052d1384266a3da0a4d16b644d7f9c4c2bfec4855720ac988a9407aebc06a3d8",
    strip_prefix = "abseil-cpp-20230802.3",
    urls = [
        "https://github.com/abseil/abseil-cpp/archive/refs/tags/20230802.3.tar.gz",
    ],
)

# ================
# Cutlass setup
# ================

git_repository(
    name = "cutlass",
    commit = "bbe579a9e3beb6ea6626d9227ec32d0dae119a49", # v3.4.1
    remote = "https://github.com/NVIDIA/cutlass",
    build_file_content = """
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cutlass",
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.hpp",
        "include/**/*.cuh",
        "tools/util/include/**/*.h",
        "tools/util/include/**/*.hpp",
        "tools/util/include/**/*.cuh",
    ]),
    includes = ["include", "tools/util/include"],
)
""",
 )
