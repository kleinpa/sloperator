#pragma once
// chatbot_lib.h — pure audio/text helpers (no I/O, no PJSIP), unit-tested.

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

// G.722: 16 kHz audio (RTP clock says 8000; PJSIP handles that).
constexpr int kAudioRate = 16000, kFrameMs = 20,
              kFrameSamples = kAudioRate * kFrameMs / 1000;

// ── Audio ─────────────────────────────────────────────────────────────────────

// Hann-windowed sinc resampler; low-passes at min(from, to) / 2, unity DC gain.
inline std::vector<int16_t> Resample(const std::vector<int16_t>& in, int from,
                                     int to) {
  if (from == to || in.empty()) return in;
  const double r = double(to) / from, fc = std::min(1.0, r), W = 16;
  const long n = in.size(), hw = long(W / fc);
  std::vector<int16_t> out(size_t(n * r));
  for (size_t j = 0; j < out.size(); j++) {
    double t = j / r, acc = 0, sum = 0;
    for (long i = std::max(0L, long(t) - hw + 1); i <= std::min(n - 1, long(t) + hw); i++) {
      double x = (t - i) * fc;
      if (std::abs(x) >= W) continue;
      double k = (x ? std::sin(M_PI * x) / (M_PI * x) : 1) * (1 + std::cos(M_PI * x / W));
      acc += in[i] * k, sum += k;
    }
    out[j] = int16_t(std::clamp(std::lround(acc / sum), -32768L, 32767L));
  }
  return out;
}

// Scales to `rms`, capped so peaks stay below ~-1 dBFS.
inline void Normalize(std::vector<int16_t>& v, double rms) {
  double s = 0;
  int pk = 1;
  for (int x : v) s += double(x) * x, pk = std::max(pk, std::abs(x));
  if (!s) return;
  double g = std::min(rms / std::sqrt(s / v.size()), 29000.0 / pk);
  for (auto& x : v) x = int16_t(std::lround(x * g));
}

inline double RmsEnergy(const int16_t* p, int n) {
  double s = 0;
  for (int i = 0; i < n; i++) s += double(p[i]) * p[i];
  return n > 0 ? std::sqrt(s / n) : 0;
}

// ── Text ──────────────────────────────────────────────────────────────────────

// Trims whitespace/quotes; empty unless it has >= 2 alphanumerics.
inline std::string CleanSentence(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r\n\"'`*"), b = s.find_last_not_of(" \t\r\n\"'`*");
  if (a == std::string::npos) return {};
  std::string t = s.substr(a, b - a + 1);
  return std::count_if(t.begin(), t.end(), [](unsigned char c) { return std::isalnum(c); }) >= 2 ? t : "";
}

// Pops complete sentences off the front of `buf`. '.' needs a lookahead char
// (so "3.5" stays whole). With `soft`, also breaks at , ; : once the clause has
// >= 4 words — gets the first audio out sooner.
inline std::vector<std::string> SplitSentences(std::string& buf, bool soft = false) {
  std::vector<std::string> out;
  size_t st = 0, words = 0;
  for (size_t i = 0; i < buf.size(); i++) {
    char c = buf[i];
    words += c == ' ' && i > st && buf[i - 1] != ' ';
    bool cut = c == '!' || c == '?' || c == '\n' || (soft && words >= 3 && (c == ',' || c == ';' || c == ':'));
    if (c == '.') {
      if (i + 1 == buf.size()) break;
      cut = !std::isdigit((unsigned char)buf[i + 1]);
    }
    if (!cut) continue;
    if (auto s = CleanSentence(buf.substr(st, i + 1 - st)); !s.empty()) out.push_back(s), soft = false;
    st = i + 1, words = 0;
  }
  buf.erase(0, st);
  return out;
}

// ── VAD ───────────────────────────────────────────────────────────────────────

struct VadParams {
  double threshold = 500;  // energy VAD: RMS of a voiced frame
  int silence_ms = 600;    // trailing silence that ends an utterance
  int min_speech_ms = 200; // shorter bursts are dropped
  std::string model;       // Silero .onnx; empty = energy VAD
  double prob = 0.5;       // Silero speech probability threshold
};

// Energy VAD. Feed 20 ms frames; returns true when `ready` holds an utterance.
struct Vad {
  VadParams p;
  std::vector<int16_t> buf, ready;
  int voiced = 0, silent = 0;

  bool ProcessFrame(const int16_t* s, int n) {
    bool v = RmsEnergy(s, n) >= p.threshold;
    if (!v && buf.empty()) return false;
    buf.insert(buf.end(), s, s + n);
    v ? (voiced++, silent = 0) : silent++;
    if (silent < p.silence_ms / kFrameMs) return false;
    bool ok = voiced >= p.min_speech_ms / kFrameMs;
    if (ok) ready = std::move(buf);
    Reset(false);
    return ok;
  }
  void Reset(bool all = true) {
    buf.clear(), voiced = silent = 0;
    if (all) ready.clear();
  }
};

// ── AudioQueue ────────────────────────────────────────────────────────────────

// Thread-safe sample FIFO: pipeline pushes TTS PCM, the audio port pops frames.
class AudioQueue {
 public:
  void Push(const std::vector<int16_t>& v) {
    std::lock_guard l(mu_);
    q_.insert(q_.end(), v.begin(), v.end());
  }
  // Fills exactly n samples, zero-padding on underflow.
  void Pop(int16_t* out, size_t n) {
    std::lock_guard l(mu_);
    size_t k = std::min(n, q_.size());
    std::copy_n(q_.begin(), k, out), q_.erase(q_.begin(), q_.begin() + k);
    std::fill(out + k, out + n, 0);
  }
  size_t Size() const { std::lock_guard l(mu_); return q_.size(); }
  void Clear() { std::lock_guard l(mu_); q_.clear(); }

 private:
  mutable std::mutex mu_;
  std::deque<int16_t> q_;
};
