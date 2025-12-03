#!/usr/bin/env python3

import logging
import os
import pathlib
import platform

import tensorflow as tf

from packaging.version import Version

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BAZELRC = os.path.join(_SCRIPT_DIR, ".bazelrc")


def _valid_vazel_version(tf_version):
    if Version(tf_version) >= Version("2.15.0"):
        target_bazel = "6.1.1"
        logging.info(
            "To ensure code compatibility with Bazel rules_foreign_cc component, "
            "we specify Bazel version greater than 6.1.1 "
            "for Tensorflow versions greater than 2.15.0."
        )
        return target_bazel
    else:
        raise ValueError(f"Unsupport TensorFlow version {tf_version}.")


def _write(line):
    with open(_BAZELRC, "a") as f:
        f.write(line + "\n")


def _write_action_env(var_name, var):
    _write(f'build --action_env {var_name}="{var}"')


def _write_clang_tidy_config():
    _write("build:clang-tidy --aspects @bazel_clang_tidy//clang_tidy:clang_tidy.bzl%clang_tidy_aspect")
    _write("build:clang-tidy --@bazel_clang_tidy//:clang_tidy_config=//:clang_tidy_config")
    _write("build:clang-tidy --@bazel_clang_tidy//:clang_tidy_executable=//:clang_tidy_executable")
    _write("build:clang-tidy --output_groups=report")


def _is_macos():
    return platform.system() == "Darwin"


def _is_linux():
    return platform.system() == "Linux"


def _is_arm64():
    return platform.machine() == "arm64"


def _get_cpp_version():
    return "c++17"


def _get_tf_header_dir():
    return tf.sysconfig.get_compile_flags()[0][2:]


def _get_tf_shared_lib_dir():
    return tf.sysconfig.get_link_flags()[0][2:]


# Converts the linkflag namespec to the full shared library name
def _get_shared_lib_name():
    namespec = tf.sysconfig.get_link_flags()
    if _is_macos():
        # MacOS
        return "lib" + namespec[1][2:] + ".dylib"
    else:
        # Linux
        return namespec[1][3:]


def _get_tf_version_integer():
    """
    Get Tensorflow version as a 4 digits string.

    For example:
      1.15.2 get 1152
      2.4.1 get 2041
      2.6.3 get 2063
      2.8.3 get 2083
      2.15.1 get 2151

    The 4-digits-string will be passed to C macro to discriminate different
    Tensorflow versions.

    We assume that major version has 1 digit, minor version has 2 digits. And
    patch version has 1 digit.
    """
    try:
        version = tf.__version__
    except AttributeError:
        raise ImportError(
            "\nPlease install a TensorFlow on your compiling machine, "
            "The compiler needs to know the version of Tensorflow "
            "and get TF c++ headers according to the installed TensorFlow. "
            "\nNote: Only TensorFlow 2.16.2 2.15.1 2.8.3, 2.6.3, 2.4.1, 1.15.2 are supported."
        )
    try:
        major, minor, patch = version.split(".")
        assert len(major) == 1, f"Tensorflow major version must be length of 1. Version: {version}"
        assert len(minor) <= 2, f"Tensorflow minor version must be less or equal to 2. Version: {version}"
        assert len(patch) == 1, f"Tensorflow patch version must be length of 1. Version: {version}"
    except Exception:
        raise ValueError(f"got wrong tf.__version__: {version}")
    tf_version_num = str(int(major) * 1000 + int(minor) * 10 + int(patch))
    if len(tf_version_num) != 4:
        raise ValueError(
            "Tensorflow version flag must be length of 4 (major"
            " version: 1, minor version: 2, patch_version: 1). But"
            " get: {}".format(tf_version_num)
        )
    return int(tf_version_num)


def _get_installed_and_valid_bazel_version():
    stream = os.popen(os.path.join(_SCRIPT_DIR, "bazel") + " version | grep label")
    output = stream.read()
    installed_bazel_version = str(output).split(":")[1].strip()
    valid_bazel_version = _valid_vazel_version(tf.__version__)
    return installed_bazel_version, valid_bazel_version


def _check_bazel_version(is_macos_arm64: bool = False):
    installed_bazel_version, valid_bazel_version = _get_installed_and_valid_bazel_version()
    if Version(installed_bazel_version) < Version(valid_bazel_version):
        raise ValueError(f"Bazel version is {installed_bazel_version}, but {valid_bazel_version} is needed.")


def _create_build_configuration():
    assert _is_linux() or _is_macos(), "Only Linux and MacOS are supported."

    print()
    print("Configuring TensorFlow Flash Attention to be built from source...")

    if os.path.isfile(_BAZELRC):
        os.remove(_BAZELRC)
    if _is_linux():
        _check_bazel_version()
    if _is_macos() and _is_arm64():
        _check_bazel_version(is_macos_arm64=True)
    logging.disable(logging.WARNING)

    _write("common --enable_bzlmod=false")
    _write("common --enable_workspace=true")

    _write_action_env("TF_HEADER_DIR", _get_tf_header_dir())
    _write_action_env("TF_SHARED_LIBRARY_DIR", _get_tf_shared_lib_dir())
    _write_action_env("TF_SHARED_LIBRARY_NAME", _get_shared_lib_name())
    _write_action_env("TF_CXX11_ABI_FLAG", tf.sysconfig.CXX11_ABI_FLAG)
    tf_cxx_standard_compile_flags = [flag for flag in tf.sysconfig.get_compile_flags() if "-std=" in flag]
    if len(tf_cxx_standard_compile_flags) > 0:
        tf_cxx_standard_compile_flag = tf_cxx_standard_compile_flags[-1]
    else:
        tf_cxx_standard_compile_flag = None
    if tf_cxx_standard_compile_flag is None:
        tf_cxx_standard = _get_cpp_version()
    else:
        tf_cxx_standard = tf_cxx_standard_compile_flag.split("-std=")[-1]
    _write_action_env("TF_CXX_STANDARD", tf_cxx_standard)

    tf_version_integer = _get_tf_version_integer()
    # This is used to trace the difference between Tensorflow versions.
    _write_action_env("TF_VERSION_INTEGER", tf_version_integer)

    _write_action_env("FOR_TF_SERVING", os.getenv("FOR_TF_SERVING", "0"))

    _write("build --spawn_strategy=standalone")
    _write("build --strategy=Genrule=standalone")
    _write("build -c opt")

    if _is_macos() or _is_linux():
        if not _is_arm64():
            _write("build --copt=-mavx")

    # _write_clang_tidy_config()

    print()
    print("Build configurations successfully written to", _BAZELRC, ":\n")
    print(pathlib.Path(_BAZELRC).read_text())


if __name__ == "__main__":
    _create_build_configuration()
