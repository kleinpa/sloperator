// test_sherpa.cc — in-process sherpa-onnx models against real model files.

#include <gtest/gtest.h>

#include <filesystem>

#include "bot_core.h"

static std::string Runfile(const char* env) { return std::string(getenv("TEST_SRCDIR")) + "/" + getenv(env); }

// Utterances found when streaming `pcm` in 20 ms frames, then 1 s of silence.
static int Utterances(VadEngine& v, std::vector<int16_t> pcm) {
  pcm.resize(pcm.size() + kAudioRate);
  std::vector<int16_t> utt;
  int n = 0;
  for (size_t i = 0; i + kFrameSamples <= pcm.size(); i += kFrameSamples) n += v.Push(&pcm[i], kFrameSamples, utt);
  return n;
}

static std::vector<int16_t> Tone(int ms) {
  std::vector<int16_t> v(ms * kAudioRate / 1000);
  for (size_t i = 0; i < v.size(); i++) v[i] = int16_t(8000 * std::sin(2 * M_PI * 440 * i / kAudioRate));
  return v;
}

TEST(SileroVad, FindsSpeechIgnoresTones) {
  VadParams p;
  p.model = Runfile("VAD_MODEL");
  EXPECT_GE(Utterances(*MakeSileroVad(p), ReadWav(Runfile("SPEECH_WAV"))), 3);
  EXPECT_EQ(Utterances(*MakeSileroVad(p), Tone(2000)), 0);
  g_cfg = {};  // energy VAD, for contrast: a tone opens a turn
  EXPECT_EQ(Utterances(*MakeVad(), Tone(2000)), 1);
}

static std::string Dir(const char* env) { return std::filesystem::path(Runfile(env)).parent_path(); }

TEST(SherpaAsr, TranscribesSpeech) {  // JFK clip
  g_cfg = {};
  auto t = MakeSherpaAsr(Dir("STT_MODEL"))->Transcribe(ReadWav(Runfile("STT_WAV")));
  EXPECT_EQ(t, "Ask not what your country can do for you. Ask what you can do for your country.");
}
