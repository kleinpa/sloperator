# Prebuilt sherpa-onnx (+ bundled onnxruntime), linux x64. The .so files have
# RPATH=$ORIGIN and share one solib dir, so they find each other at runtime.
load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")

[cc_import(
    name = n,
    shared_library = "lib/lib%s.so" % n,
) for n in ["onnxruntime", "sherpa-onnx-c-api", "sherpa-onnx-cxx-api"]]

cc_library(
    name = "sherpa_onnx",
    hdrs = glob(["include/**/*.h"]),
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
    deps = [":onnxruntime", ":sherpa-onnx-c-api", ":sherpa-onnx-cxx-api"],
)
