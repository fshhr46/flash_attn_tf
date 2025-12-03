load(
    "@local_config_tf//:build_defs.bzl",
    "DTF_VERSION_INTEGER",
    "D_GLIBCXX_USE_CXX11_ABI",
    "FOR_TF_SERVING",
    "TF_CXX_STANDARD",
)

def custom_op_library(
        name,
        srcs = [],
        deps = [],
        copts = [],
        **kwargs):
    if FOR_TF_SERVING == "1":
        deps = deps + [
            "@local_config_tf//:tf_header_lib",
        ]
    else:
        deps = deps + [
            "@local_config_tf//:libtensorflow_framework",
            "@local_config_tf//:tf_header_lib",
        ]

    final_copts = copts + [
        "-pthread",
        "-funroll-loops",
        D_GLIBCXX_USE_CXX11_ABI,
        DTF_VERSION_INTEGER,
        "-std=" + TF_CXX_STANDARD,
        "-stdlib=libstdc++",
    ]

    final_linkopts = [
        "-stdlib=libstdc++",
    ]

    if FOR_TF_SERVING == "1":
        native.cc_library(
            name = name,
            srcs = srcs,
            copts = final_copts,
            linkopts = final_linkopts,
            alwayslink = 1,
            deps = deps,
            **kwargs
        )
    else:
        native.cc_binary(
            name = name,
            srcs = srcs,
            copts = final_copts,
            linkopts = final_linkopts,
            linkshared = 1,
            deps = deps,
            **kwargs
        )
