// test_chatbot.cc — unit tests for chatbot_lib.h.

#include "chatbot_lib.h"

#include <gtest/gtest.h>

static std::vector<int16_t> Sine(int n, int rate, double hz, double amp = 8000) {
  std::vector<int16_t> v(n);
  for (int i = 0; i < n; i++) v[i] = int16_t(amp * std::sin(2 * M_PI * hz * i / rate));
  return v;
}
static int ZeroCrossings(const std::vector<int16_t>& v) {
  int z = 0;
  for (size_t i = 1; i < v.size(); i++) z += (v[i - 1] < 0) != (v[i] < 0);
  return z;
}

// Regression: 22050→16000 used to step by int(1.378)=1, i.e. no decimation —
// playback ran at 73% speed, ~5 semitones low.
TEST(Resample, KeepsPitchAndDuration) {
  auto in = Sine(22050, 22050, 440);  // 1 s
  auto out = Resample(in, 22050, 16000);
  EXPECT_EQ(out.size(), 16000u);
  EXPECT_NEAR(ZeroCrossings(out), 880, 4);
}

TEST(Resample, RemovesAliasesWhenDownsampling) {
  // 10 kHz is above the 8 kHz output Nyquist: should be filtered, not folded.
  auto out = Resample(Sine(24000, 24000, 10000), 24000, 16000);
  EXPECT_LT(RmsEnergy(out.data() + 100, out.size() - 200), 8000 / std::sqrt(2) * 0.05);
}

TEST(Resample, UnityGainAndPassthrough) {
  auto out = Resample(std::vector<int16_t>(2205, 1000), 22050, 16000);
  for (size_t i = 0; i < out.size(); i++) ASSERT_EQ(out[i], 1000) << i;
  auto same = Sine(100, 16000, 440);
  EXPECT_EQ(Resample(same, 16000, 16000), same);
}

TEST(Normalize, HitsTargetRmsWithPeakCap) {
  auto quiet = Sine(16000, 16000, 300, 500);
  Normalize(quiet, 4000);
  EXPECT_NEAR(RmsEnergy(quiet.data(), quiet.size()), 4000, 20);

  std::vector<int16_t> spiky(1000, 10);
  spiky[500] = 1000;  // crest factor too high for the RMS target
  Normalize(spiky, 4000);
  EXPECT_LE(*std::max_element(spiky.begin(), spiky.end()), 29000);

  std::vector<int16_t> silence(100, 0);
  Normalize(silence, 4000);
  EXPECT_EQ(silence, std::vector<int16_t>(100, 0));
}

TEST(SplitSentences, StreamsCompleteSentencesOnly) {
  std::string buf;
  std::vector<std::string> got;
  for (const char* tok : {"Hi", " there", ". It", " costs", " 3", ".", "5 dollars", "! Bye"})
    for (auto& s : SplitSentences(buf += tok)) got.push_back(s);
  EXPECT_EQ(got, (std::vector<std::string>{"Hi there.", "It costs 3.5 dollars!"}));
  EXPECT_EQ(buf, " Bye");
}

TEST(SplitSentences, SoftBreakOnlyForLongClauses) {
  std::string a = "Sure, one moment";
  EXPECT_TRUE(SplitSentences(a, true).empty());  // "Sure," is too short to send alone
  std::string b = "Sure thing, let me check that, okay";
  EXPECT_EQ(SplitSentences(b, true), std::vector<std::string>{"Sure thing, let me check that,"});
  std::string c = "Sure thing, let me check that, okay";
  EXPECT_TRUE(SplitSentences(c).empty());
}

TEST(CleanSentence, TrimsAndRejectsJunk) {
  EXPECT_EQ(CleanSentence("  \"**Hello.**\"\n"), "Hello.");
  EXPECT_EQ(CleanSentence(" . "), "");
  EXPECT_EQ(CleanSentence("\"A\""), "");
}

TEST(Vad, DetectsUtteranceAndDropsBlips) {
  Vad v;
  v.p = {500, 100, 100};  // 5 silent frames end it; >= 5 voiced frames needed
  auto loud = Sine(kFrameSamples, kAudioRate, 440), quiet = std::vector<int16_t>(kFrameSamples);
  auto feed = [&](auto& f, int n) {
    bool done = false;
    for (int i = 0; i < n; i++) done |= v.ProcessFrame(f.data(), f.size());
    return done;
  };
  EXPECT_FALSE(feed(quiet, 20));            // silence alone does nothing
  EXPECT_FALSE(feed(loud, 2) || feed(quiet, 10));  // too short: dropped
  EXPECT_FALSE(feed(loud, 6));
  EXPECT_TRUE(feed(quiet, 5));
  EXPECT_EQ(v.ready.size(), size_t(11 * kFrameSamples));  // speech + trailing silence
}

TEST(AudioQueue, PopsInOrderAndPadsWithSilence) {
  AudioQueue q;
  q.Push({1, 2, 3});
  int16_t out[5];
  q.Pop(out, 2);
  EXPECT_EQ(out[0], 1);
  EXPECT_EQ(out[1], 2);
  q.Pop(out, 5);
  EXPECT_EQ(std::vector<int16_t>(out, out + 5), (std::vector<int16_t>{3, 0, 0, 0, 0}));
  EXPECT_EQ(q.Size(), 0u);
}
