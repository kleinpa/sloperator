// sherpa.cc — in-process models via sherpa-onnx's C API (its C++ API .so is
// built with the old std::string ABI, so it can't be mixed with ours).

#include "absl/log/log.h"
#include "bot_core.h"
#include "sherpa-onnx/c-api/c-api.h"

template <auto F> struct Free { template <class T> void operator()(T* p) const { F(p); } };
template <class T, auto F> using Ptr = std::unique_ptr<const T, Free<F>>;

static std::vector<float> F32(const int16_t* s, int n) {
  std::vector<float> f(s, s + n);
  for (auto& x : f) x /= 32768;
  return f;
}
static std::vector<int16_t> I16(const float* f, int n) {
  std::vector<int16_t> s(n);
  for (int i = 0; i < n; i++) s[i] = int16_t(std::clamp(f[i] * 32767, -32768.f, 32767.f));
  return s;
}

// Silero: neural speech probability, so tones/noise/music don't open a turn.
struct SileroVad : VadEngine {
  Ptr<SherpaOnnxVoiceActivityDetector, SherpaOnnxDestroyVoiceActivityDetector> d;
  explicit SileroVad(const VadParams& p) {
    SherpaOnnxVadModelConfig c{};
    c.silero_vad = {p.model.c_str(), float(p.prob), p.silence_ms / 1000.f, p.min_speech_ms / 1000.f, 512, 20};
    c.sample_rate = kAudioRate, c.num_threads = 1;
    d.reset(SherpaOnnxCreateVoiceActivityDetector(&c, 30));
    if (!d) LOG(FATAL) << "[vad] can't load " << p.model;
  }
  bool Push(const int16_t* s, int n, std::vector<int16_t>& utt) override {
    auto f = F32(s, n);
    SherpaOnnxVoiceActivityDetectorAcceptWaveform(d.get(), f.data(), n);
    if (SherpaOnnxVoiceActivityDetectorEmpty(d.get())) return false;
    Ptr<SherpaOnnxSpeechSegment, SherpaOnnxDestroySpeechSegment> seg(SherpaOnnxVoiceActivityDetectorFront(d.get()));
    utt = I16(seg->samples, seg->n);
    SherpaOnnxVoiceActivityDetectorPop(d.get());
    return true;
  }
  void Reset() override { SherpaOnnxVoiceActivityDetectorReset(d.get()); }
};

std::unique_ptr<VadEngine> MakeSileroVad(const VadParams& p) { return std::make_unique<SileroVad>(p); }

std::vector<int16_t> ReadWav(const std::string& path) {
  Ptr<SherpaOnnxWave, SherpaOnnxFreeWave> w(SherpaOnnxReadWave(path.c_str()));
  return w ? Resample(I16(w->samples, w->num_samples), w->sample_rate, kAudioRate) : std::vector<int16_t>{};
}
