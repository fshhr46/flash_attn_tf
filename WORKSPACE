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
# pre-tensorflow setup
# ================

# Note: zlib is placed earlier as tensorflow's zlib does not include unzip
# referred by arrow
http_archive(
    name = "zlib",
    build_file = "//third_party:zlib.BUILD",
    patch_cmds = ["""sed -i.bak '29i\\'$'\\n#include<zconf.h>\\n' contrib/minizip/crypt.h"""],
    sha256 = "b3a24de97a8fdbc835b9833169501030b8977031bcb54b3b3ac13740f846ab30",
    strip_prefix = "zlib-1.2.13",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/zlib.net/zlib-1.2.13.tar.gz",
        "https://zlib.net/zlib-1.2.13.tar.gz",
    ],
)

# Note: snappy is placed earlier as tensorflow's snappy does not include snappy-c
# referred by arrow
http_archive(
    name = "snappy",
    build_file = "//third_party:snappy.BUILD",
    sha256 = "16b677f07832a612b0836178db7f374e414f94657c138e6993cbfc5dcc58651f",
    strip_prefix = "snappy-1.1.8",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/snappy/archive/1.1.8.tar.gz",
        "https://github.com/google/snappy/archive/1.1.8.tar.gz",
    ],
)

# Note: boringssl is placed earlier as boringssl needs to be patched for mongodb
# referred by arrow
http_archive(
    name = "boringssl",
    patch_args = ["-p1"],
    patches = ["//third_party:boringssl.patch"],
    sha256 = "a9c3b03657d507975a32732f04563132b4553c20747cec6dc04de475c8bdf29f",
    strip_prefix = "boringssl-80ca9f9f6ece29ab132cce4cf807a9465a18cfac",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/boringssl/archive/80ca9f9f6ece29ab132cce4cf807a9465a18cfac.tar.gz",
        "https://github.com/google/boringssl/archive/80ca9f9f6ece29ab132cce4cf807a9465a18cfac.tar.gz",
    ],
)

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

http_archive(
    name = "com_google_googletest",
    strip_prefix = "googletest-531e29485c3bd21a5f2fa846dfc62f9c38756033",
    urls = ["https://github.com/google/googletest/archive/531e29485c3bd21a5f2fa846dfc62f9c38756033.zip"],
)

# ==================
# Other dependencies
# ==================

# oneTBB is a C++ template library for parallel programming.
git_repository(
    name = "oneTBB",
    commit = "45587e94dfb6dfe00220c5f520020a5bc745e92f",  # v2022.1.0
    remote = "https://github.com/uxlfoundation/oneTBB",
)

git_repository(
    name = "re2",
    build_file = "//third_party:re2.BUILD",
    commit = "6dcd83d60f7944926bfd308cc13979fc53dd69ca",  # 2024-07-02
    remote = "https://github.com/google/re2",
)

http_archive(
    name = "arrow",
    build_file = "//third_party:arrow.BUILD",
    patch_cmds = [
        """sed -i.bak '24i\\'$'\\n#undef ARROW_WITH_OPENTELEMETRY\\n' cpp/src/arrow/util/tracing_internal.h""",
    ],
    sha256 = "19ece12de48e51ce4287d2dee00dc358fbc5ff02f41629d16076f77b8579e272",
    strip_prefix = "arrow-apache-arrow-8.0.0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/apache/arrow/archive/apache-arrow-8.0.0.tar.gz",
        "https://github.com/apache/arrow/archive/apache-arrow-8.0.0.tar.gz",
    ],
)

# referred by thrift
http_archive(
    name = "boost",
    build_file = "//third_party:boost.BUILD",
    sha256 = "c66e88d5786f2ca4dbebb14e06b566fb642a1a6947ad8cc9091f9f445134143f",
    strip_prefix = "boost_1_72_0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/dl.bintray.com/boostorg/release/1.72.0/source/boost_1_72_0.tar.gz",
        "https://storage.googleapis.com/mirror.tensorflow.org/downloads.sourceforge.net/project/boost/boost/1.72.0/boost_1_72_0.tar.gz",
        "https://dl.bintray.com/boostorg/release/1.72.0/source/boost_1_72_0.tar.gz",
        "https://downloads.sourceforge.net/project/boost/boost/1.72.0/boost_1_72_0.tar.gz",
    ],
)

# referred by arrow
http_archive(
    name = "brotli",
    build_file = "//third_party:brotli.BUILD",
    sha256 = "4c61bfb0faca87219ea587326c467b95acb25555b53d1a421ffa3c8a9296ee2c",
    strip_prefix = "brotli-1.0.7",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/brotli/archive/v1.0.7.tar.gz",
        "https://github.com/google/brotli/archive/v1.0.7.tar.gz",
    ],
)

# referred by arrow
http_archive(
    name = "bzip2",
    build_file = "//third_party:bzip2.BUILD",
    sha256 = "db106b740252669664fd8f3a1c69fe7f689d5cd4b132f82ba82b9afba27627df",
    strip_prefix = "bzip2-bzip2-1.0.8",
    urls = [
        "https://gitlab.com/bzip2/bzip2/-/archive/bzip2-1.0.8/bzip2-bzip2-1.0.8.tar.gz",
    ],
)

# referred by arrow
http_archive(
    name = "double-conversion",
    sha256 = "a63ecb93182134ba4293fd5f22d6e08ca417caafa244afaa751cbfddf6415b13",
    strip_prefix = "double-conversion-3.1.5",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/double-conversion/archive/v3.1.5.tar.gz",
        "https://github.com/google/double-conversion/archive/v3.1.5.tar.gz",
    ],
)

# referred by arrow
http_archive(
    name = "lz4",
    build_file = "//third_party:lz4.BUILD",
    patch_cmds = [
        """sed -i.bak 's/__attribute__ ((__visibility__ ("default")))//g' lib/lz4frame.h """,
    ],
    sha256 = "658ba6191fa44c92280d4aa2c271b0f4fbc0e34d249578dd05e50e76d0e5efcc",
    strip_prefix = "lz4-1.9.2",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/lz4/lz4/archive/v1.9.2.tar.gz",
        "https://github.com/lz4/lz4/archive/v1.9.2.tar.gz",
    ],
)

# referred by arrow
http_archive(
    name = "thrift",
    build_file = "//third_party:thrift.BUILD",
    sha256 = "5da60088e60984f4f0801deeea628d193c33cec621e78c8a43a5d8c4055f7ad9",
    strip_prefix = "thrift-0.13.0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/apache/thrift/archive/v0.13.0.tar.gz",
        "https://github.com/apache/thrift/archive/v0.13.0.tar.gz",
    ],
)

# referred by boost
http_archive(
    name = "xz",
    build_file = "//third_party:xz.BUILD",
    sha256 = "0d2b89629f13dd1a0602810529327195eff5f62a0142ccd65b903bc16a4ac78a",
    strip_prefix = "xz-5.2.5",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/xz-mirror/xz/archive/v5.2.5.tar.gz",
        "https://github.com/xz-mirror/xz/archive/v5.2.5.tar.gz",
    ],
)

# referred by arrow
http_archive(
    name = "xsimd",
    build_file = "//third_party:xsimd.BUILD",
    sha256 = "21b4700e9ef70f6c9a86952047efd8272317df4e6fee35963de9394fd9c5677f",
    strip_prefix = "xsimd-8.0.1",
    urls = [
        "https://github.com/xtensor-stack/xsimd/archive/refs/tags/8.0.1.tar.gz",
    ],
)

# referred by arrow
http_archive(
    name = "zstd",
    build_file = "//third_party:zstd.BUILD",
    sha256 = "a364f5162c7d1a455cc915e8e3cf5f4bd8b75d09bc0f53965b0c9ca1383c52c8",
    strip_prefix = "zstd-1.4.4",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/facebook/zstd/archive/v1.4.4.tar.gz",
        "https://github.com/facebook/zstd/archive/v1.4.4.tar.gz",
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