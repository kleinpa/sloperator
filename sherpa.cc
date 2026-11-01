// sherpa.cc — in-process VAD/STT/TTS via sherpa-onnx's C API (its C++ API .so is
// built with the old std::string ABI, so it can't be mixed with ours).

#include <filesystem>

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

// First file in `dir` named stem*ext ("" if none).
static std::string Find(const std::string& dir, const std::string& stem, const std::string& ext = "") {
  for (auto& e : std::filesystem::directory_iterator(dir))
    if (auto n = e.path().filename().string();
        n.rfind(stem, 0) == 0 && n.size() >= ext.size() && n.compare(n.size() - ext.size(), ext.size(), ext) == 0)
      return e.path();
  return "";
}

// Models load once per dir and are shared by all calls (calls serialize on mu).
template <class M> static std::shared_ptr<M> Load(const std::string& dir) {
  static std::mutex mu;
  static std::map<std::string, std::shared_ptr<M>> loaded;
  std::lock_guard l(mu);
  auto& m = loaded[dir];
  return m ? m : m = std::make_shared<M>(dir);
}

// Offline ASR; the model kind is inferred from the files in `dir`:
// joiner* → NeMo transducer (Parakeet), decoder_model_merged* → Moonshine v2.
struct AsrModel {
  Ptr<SherpaOnnxOfflineRecognizer, SherpaOnnxDestroyOfflineRecognizer> r;
  std::mutex mu;
  explicit AsrModel(const std::string& dir) {
    std::string f[] = {Find(dir, "encoder"), Find(dir, "decoder"), Find(dir, "joiner"),
                       Find(dir, "decoder_model_merged"), Find(dir, "tokens.txt")};
    SherpaOnnxOfflineRecognizerConfig c{};
    c.feat_config = {kAudioRate, 80};
    auto& m = c.model_config;
    m.tokens = f[4].c_str(), m.num_threads = g_cfg.threads, m.provider = "cpu";
    if (!f[2].empty()) m.transducer = {f[0].c_str(), f[1].c_str(), f[2].c_str()}, m.model_type = "nemo_transducer";
    else m.moonshine.encoder = f[0].c_str(), m.moonshine.merged_decoder = f[3].c_str();
    c.decoding_method = "greedy_search";
    r.reset(SherpaOnnxCreateOfflineRecognizer(&c));
    if (!r) LOG(FATAL) << "[stt] can't load " << dir;
  }
};
struct SherpaAsr : AsrClient {
  std::shared_ptr<AsrModel> m;
  std::string Transcribe(const std::vector<int16_t>& pcm) override {
    std::lock_guard l(m->mu);
    Ptr<SherpaOnnxOfflineStream, SherpaOnnxDestroyOfflineStream> s(SherpaOnnxCreateOfflineStream(m->r.get()));
    auto f = F32(pcm.data(), pcm.size());
    SherpaOnnxAcceptWaveformOffline(s.get(), kAudioRate, f.data(), f.size());
    SherpaOnnxDecodeOfflineStream(m->r.get(), s.get());
    Ptr<SherpaOnnxOfflineRecognizerResult, SherpaOnnxDestroyOfflineRecognizerResult> res(
        SherpaOnnxGetOfflineStreamResult(s.get()));
    std::string t = res->text;
    return t.erase(0, t.find_first_not_of(' '));
  }
};

// Offline TTS: voices.bin → Kokoro, else Piper (VITS). Both use espeak-ng-data.
struct TtsModel {
  Ptr<SherpaOnnxOfflineTts, SherpaOnnxDestroyOfflineTts> t;
  std::mutex mu;
  explicit TtsModel(const std::string& dir) {
    std::string voices = Find(dir, "voices.bin"), model = Find(dir, voices.empty() ? "" : "model", ".onnx"),
                tokens = Find(dir, "tokens.txt"), data = dir + "/espeak-ng-data";
    SherpaOnnxOfflineTtsConfig c{};
    auto& m = c.model;
    m.num_threads = g_cfg.threads, m.provider = "cpu";
    if (voices.empty()) m.vits = {model.c_str(), "", tokens.c_str(), data.c_str(), 0.667f, 0.8f, 1, ""};
    else m.kokoro = {model.c_str(), voices.c_str(), tokens.c_str(), data.c_str(), 1, "", "", ""};
    c.max_num_sentences = 1, c.silence_scale = 0.2f;
    t.reset(SherpaOnnxCreateOfflineTts(&c));
    if (!t) LOG(FATAL) << "[tts] can't load " << dir;
  }
};
struct SherpaTts : TtsClient {
  std::shared_ptr<TtsModel> m;
  std::vector<int16_t> Synthesize(const std::string& text) override {
    std::lock_guard l(m->mu);
    SherpaOnnxGenerationConfig g{};
    g.silence_scale = 0.2f, g.speed = g_cfg.tts_speed, g.sid = g_cfg.tts_speaker;
    Ptr<SherpaOnnxGeneratedAudio, SherpaOnnxDestroyOfflineTtsGeneratedAudio> a(
        SherpaOnnxOfflineTtsGenerateWithConfig(m->t.get(), text.c_str(), &g, nullptr, nullptr));
    return a ? Resample(I16(a->samples, a->n), a->sample_rate, kAudioRate) : std::vector<int16_t>{};
  }
};

std::unique_ptr<AsrClient> MakeSherpaAsr(const std::string& dir) {
  auto c = std::make_unique<SherpaAsr>();
  c->m = Load<AsrModel>(dir);
  return c;
}
std::unique_ptr<TtsClient> MakeSherpaTts(const std::string& dir) {
  auto c = std::make_unique<SherpaTts>();
  c->m = Load<TtsModel>(dir);
  return c;
}

std::vector<int16_t> ReadWav(const std::string& path) {
  Ptr<SherpaOnnxWave, SherpaOnnxFreeWave> w(SherpaOnnxReadWave(path.c_str()));
  return w ? Resample(I16(w->samples, w->num_samples), w->sample_rate, kAudioRate) : std::vector<int16_t>{};
}
