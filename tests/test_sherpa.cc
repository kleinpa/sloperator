// test_sherpa.cc — prebuilt sherpa-onnx links, loads, and runs onnxruntime-free code.

#include <gtest/gtest.h>

#include "sherpa-onnx/c-api/cxx-api.h"

TEST(Sherpa, LoadsAndResamples) {
  EXPECT_STREQ(SherpaOnnxGetVersionStr(), "2.0.1");
  auto r = sherpa_onnx::cxx::LinearResampler::Create(22050, 16000, 0.99f * 8000, 6);
  std::vector<float> in(22050, 0.5f);
  EXPECT_NEAR(r.Resample(in.data(), in.size(), true).size(), 16000u, 2);
}
