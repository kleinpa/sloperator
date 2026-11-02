// bot_core.cc — Wyoming STT/TTS + Ollama clients, Session pipeline, PJSIP glue.

#include "bot_core.h"

#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
#include <yaml-cpp/yaml.h>

#include <chrono>
#include <variant>

#include "absl/log/log.h"

#define CPPHTTPLIB_NO_EXCEPTIONS
#include "httplib.h"

using json = nlohmann::json;

Config g_cfg;
std::vector<int16_t> g_greeting_pcm;

// ── Config ────────────────────────────────────────────────────────────────────

// Every config key ↔ Config field; dotted keys nest. LoadConfig, DumpConfig and
// ConfigKeys all come from this one table, so nothing is settable in code only.
using Field = std::variant<std::string*, int*, double*, bool*, std::map<std::string, std::string>*, json*>;
static std::vector<std::pair<std::string, Field>> Fields(Config& c) {
  return {{"whisper", &c.whisper}, {"ollama", &c.ollama}, {"piper", &c.piper}, {"pbx", &c.pbx},
          {"model", &c.model}, {"think", &c.think}, {"options", &c.options},
          {"system_prompt", &c.prompt}, {"knowledge", &c.knowledge}, {"directory", &c.directory},
          {"greeting", &c.greeting}, {"fallback", &c.fallback}, {"language", &c.language},
          {"stt_model", &c.stt_model}, {"tts_model", &c.tts_model}, {"tts_speed", &c.tts_speed},
          {"tts_varispeed", &c.tts_varispeed}, {"tts_speaker", &c.tts_speaker}, {"tts_rms", &c.tts_rms},
          {"threads", &c.threads}, {"max_calls", &c.max_calls}, {"history_turns", &c.history_turns},
          {"vad.model", &c.vad.model}, {"vad.prob", &c.vad.prob}, {"vad.threshold", &c.vad.threshold},
          {"vad.silence_ms", &c.vad.silence_ms}, {"vad.min_speech_ms", &c.vad.min_speech_ms},
          {"vad.max_speech_s", &c.vad.max_speech_s}, {"knowledge_intro", &c.knowledge_intro},
          {"transfer_intro", &c.transfer_intro}, {"transfer_say", &c.transfer_say}, {"cues", &c.cues},
          {"keep_alive", &c.keep_alive}, {"llm_timeout_s", &c.llm_timeout_s},
          {"service_timeout_s", &c.service_timeout_s}, {"chunk_first_words", &c.chunk_first_words},
          {"chunk_words", &c.chunk_words}, {"cue_db", &c.cue_db}};
}
static json::json_pointer Ptr(std::string k) { return json::json_pointer("/" + (std::replace(k.begin(), k.end(), '.', '/'), k)); }

// YAML → JSON; quoted scalars stay strings ("0042"), plain ones get typed.
static json ToJson(const YAML::Node& n) {
  if (n.IsMap()) { json j = json::object(); for (auto kv : n) j[kv.first.as<std::string>()] = ToJson(kv.second); return j; }
  if (n.IsSequence()) { json j = json::array(); for (auto v : n) j.push_back(ToJson(v)); return j; }
  if (!n.IsScalar()) return nullptr;
  if (n.Tag() == "!") return n.as<std::string>();
  try { return n.as<long>(); } catch (...) {}
  try { return n.as<double>(); } catch (...) {}
  try { return n.as<bool>(); } catch (...) {}
  return n.as<std::string>();
}
static std::string Str(const json& v) { return v.is_string() ? v.get<std::string>() : v.dump(); }

std::vector<std::string> LoadConfig(const std::string& text) {
  json y = ToJson(YAML::Load(text));
  if (!y.is_object()) y = json::object();
  auto fields = Fields(g_cfg);
  for (auto& [k, f] : fields)
    if (auto p = Ptr(k); y.contains(p) && !y[p].is_null())
      std::visit([&](auto* d) {
        using T = std::decay_t<decltype(*d)>;
        auto& v = y[p];
        if constexpr (std::is_same_v<T, std::string>) *d = Str(v);
        else if constexpr (std::is_same_v<T, json>) v.is_object() && d->is_object() ? d->update(v) : void(*d = v);
        else if constexpr (std::is_same_v<T, std::map<std::string, std::string>>) for (auto& [a, b] : v.items()) (*d)[a] = Str(b);
        else *d = v.get<T>();
      }, f);
  std::vector<std::string> unknown;  // typos, or keys from an older schema
  std::function<void(const json&, std::string)> walk = [&](const json& j, std::string at) {
    for (auto& [k, v] : j.items()) {
      auto key = at + k;
      if (std::any_of(fields.begin(), fields.end(), [&](auto& f) { return f.first == key; })) continue;
      if (v.is_object()) walk(v, key + ".");
      else unknown.push_back(key);
    }
  };
  walk(y, "");
  return unknown;
}

std::string DumpConfig() {
  json j;
  for (auto& [k, f] : Fields(g_cfg))
    std::visit([&, k = k](auto* d) {
      if constexpr (std::is_same_v<std::decay_t<decltype(*d)>, json>) { if (!d->is_null()) j[Ptr(k)] = *d; }
      else j[Ptr(k)] = *d;
    }, f);
  return j.dump(2);  // JSON is YAML: feeds straight back into --config_file
}

std::vector<std::string> ConfigKeys() {
  std::vector<std::string> k;
  for (auto& [n, _] : Fields(g_cfg)) k.push_back(n);
  return k;
}

std::string SystemPrompt() {
  std::string p = g_cfg.prompt;
  if (!g_cfg.knowledge.empty())
    p += "\n\n" + g_cfg.knowledge_intro + "\n" + g_cfg.knowledge;
  if (!g_cfg.directory.empty()) {
    p += "\n\n" + g_cfg.transfer_intro;
    for (auto& [d, _] : g_cfg.directory) p += " " + d + ",";
    p.back() = '.';
  }
  return p;
}

// ── Wyoming (newline JSON header + data + payload over TCP) ───────────────────

static int Connect(const std::string& addr) {
  auto c = addr.rfind(':');
  addrinfo hints{}, *res;
  hints.ai_socktype = SOCK_STREAM;
  if (getaddrinfo(addr.substr(0, c).c_str(), addr.substr(c + 1).c_str(), &hints, &res)) return -1;
  int fd = -1;
  for (auto* p = res; p && fd < 0; p = p->ai_next)
    if ((fd = socket(p->ai_family, p->ai_socktype, 0)) >= 0 && connect(fd, p->ai_addr, p->ai_addrlen))
      close(fd), fd = -1;
  freeaddrinfo(res);
  timeval tv{g_cfg.service_timeout_s, 0};  // a hung service must not hang the call
  if (fd >= 0) setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv),
               setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
  if (fd < 0) LOG(ERROR) << "[wyoming] connect failed: " << addr;
  return fd;
}

static bool IO(int fd, void* p, size_t n, bool tx) {
  for (auto* b = (char*)p; n;) {
    ssize_t r = tx ? send(fd, b, n, MSG_NOSIGNAL) : recv(fd, b, n, 0);
    if (r <= 0) return false;
    b += r, n -= r;
  }
  return true;
}

static bool Send(int fd, const std::string& type, const json& data = {},
                 const void* payload = nullptr, size_t len = 0) {
  std::string d = data.is_null() ? "" : data.dump();
  json h = {{"type", type}, {"version", "1.8.0"}};
  if (!d.empty()) h["data_length"] = d.size();
  if (len) h["payload_length"] = len;
  std::string hs = h.dump() + "\n";
  return IO(fd, hs.data(), hs.size(), true) && IO(fd, d.data(), d.size(), true) &&
         (!len || IO(fd, (void*)payload, len, true));
}

struct Msg { std::string type; json data; std::string payload; };
static Msg Recv(int fd) {
  std::string h;
  for (char c; recv(fd, &c, 1, 0) == 1 && c != '\n';) h += c;
  json j = json::parse(h, nullptr, false);
  if (j.is_discarded()) return {};
  std::string d(j.value("data_length", 0), 0);
  Msg m{j.value("type", ""), {}, std::string(j.value("payload_length", 0), 0)};
  if (!IO(fd, d.data(), d.size(), false) || !IO(fd, m.payload.data(), m.payload.size(), false)) return {};
  if (!d.empty()) m.data = json::parse(d, nullptr, false);
  if (j.contains("data")) m.data = j["data"];  // inline data (older servers)
  return m;
}

struct WyomingAsr : AsrClient {
  std::string Transcribe(const std::vector<int16_t>& pcm) override {
    int fd = Connect(g_cfg.whisper);
    if (fd < 0) return {};
    json fmt = {{"rate", kAudioRate}, {"width", 2}, {"channels", 1}};
    std::string text;
    if (Send(fd, "transcribe", {{"language", g_cfg.language}}) && Send(fd, "audio-start", fmt) &&
        Send(fd, "audio-chunk", fmt, pcm.data(), pcm.size() * 2) && Send(fd, "audio-stop"))
      for (Msg m; !(m = Recv(fd)).type.empty();)
        if (m.type == "transcript") { text = m.data.value("text", ""); break; }
        else if (m.type == "error") { LOG(ERROR) << "[stt] " << m.data; break; }
    close(fd);
    return text;
  }
};

struct WyomingTts : TtsClient {
  std::vector<int16_t> Synthesize(const std::string& text) override {
    int fd = Connect(g_cfg.piper);
    if (fd < 0) return {};
    std::vector<int16_t> pcm;
    int rate = kAudioRate;
    if (Send(fd, "synthesize", {{"text", text}}))
      for (Msg m; !(m = Recv(fd)).type.empty() && m.type != "audio-stop";)
        if (m.type == "audio-start") rate = m.data.value("rate", rate);
        else if (m.type == "audio-chunk")
          pcm.insert(pcm.end(), (int16_t*)m.payload.data(), (int16_t*)(m.payload.data() + m.payload.size()));
        else if (m.type == "error") { LOG(ERROR) << "[tts] " << m.data; break; }
    close(fd);
    return Resample(pcm, rate, kAudioRate);
  }
};

// ── Ollama /api/chat (NDJSON stream, optional transfer_call tool) ─────────────

struct Ollama : LlmClient {
  std::mutex mu;
  httplib::Client* live = nullptr;  // in-flight request, for Cancel()
  bool stopped = false;
  void Cancel() override {
    std::lock_guard l(mu);
    stopped = true;
    if (live) live->stop();  // also aborts prompt processing, before any token
  }
  LlmReply Chat(const json& messages, std::function<bool(const std::string&)> say) override {
    json req = {{"model", g_cfg.model}, {"messages", messages}, {"options", g_cfg.options},
                {"stream", true}, {"keep_alive", g_cfg.keep_alive}};
    if (!g_cfg.think.is_null()) req["think"] = g_cfg.think;
    if (!g_cfg.directory.empty()) {
      json depts = json::array();
      for (auto& [d, _] : g_cfg.directory) depts.push_back(d);
      json params = {{"type", "object"}, {"required", json::array({"department"})},
                     {"properties", {{"department", {{"type", "string"}, {"enum", depts}}}}}};
      req["tools"] = json::array({{{"type", "function"},
                                   {"function", {{"name", "transfer_call"},
                                                 {"description", "Transfer the caller to a department."},
                                                 {"parameters", params}}}}});
    }
    LlmReply r;
    std::string line, buf;
    bool go = true, first = true;
    auto emit = [&](const std::string& s) { if (go) go = say(s), first = false; };
    httplib::Client cli(g_cfg.ollama);
    cli.set_connection_timeout(5), cli.set_read_timeout(g_cfg.llm_timeout_s);
    {
      std::lock_guard l(mu);
      if (stopped) return r;
      live = &cli;
    }
    httplib::Request q;
    q.method = "POST", q.path = "/api/chat", q.body = req.dump();
    q.headers = {{"Content-Type", "application/json"}};
    q.content_receiver = [&](const char* d, size_t n, uint64_t, uint64_t) {
      for (line.append(d, n); go;) {  // NDJSON lines may straddle chunks
        auto nl = line.find('\n');
        if (nl == std::string::npos) break;
        json j = json::parse(line.substr(0, nl), nullptr, false);
        line.erase(0, nl + 1);
        if (j.is_discarded()) continue;
        if (j.contains("error")) LOG(ERROR) << "[llm] " << j["error"];
        json m = j.value("message", json::object());
        for (auto& t : m.value("tool_calls", json::array()))
          if (auto f = t.value("function", json::object()); f.value("name", "") == "transfer_call")
            r.transfer = f.value("arguments", json::object()).value("department", "");
        auto c = m.value("content", "");
        r.text += c, buf += c;
        for (auto& s : SplitSentences(buf, first ? g_cfg.chunk_first_words : g_cfg.chunk_words, g_cfg.chunk_words))
          emit(s);
      }
      return go;
    };
    httplib::Response res;
    httplib::Error err;
    bool sent = cli.send(q, res, err);
    {
      std::lock_guard l(mu);
      live = nullptr;
      if (stopped) return r;
    }
    r.status = res.status;
    if (!sent && go) LOG(ERROR) << "[llm] " << httplib::to_string(err);
    else if (res.status && res.status != 200) {
      auto e = json::parse(line.empty() ? res.body : line, nullptr, false);
      LOG(ERROR) << "[llm] HTTP " << res.status << ": " << (e.is_object() ? e.value("error", "") : line);
    }
    if (auto s = CleanSentence(buf); !s.empty()) emit(s);
    return r;
  }
};

struct EnergyVad : VadEngine {
  Vad v;
  bool Push(const int16_t* s, int n, std::vector<int16_t>& utt) override {
    return v.ProcessFrame(s, n) && (utt = std::move(v.ready), true);
  }
  void Reset() override { v.Reset(); }
};
std::unique_ptr<VadEngine> MakeVad() {
  if (!g_cfg.vad.model.empty()) return MakeSileroVad(g_cfg.vad);
  auto e = std::make_unique<EnergyVad>();
  e->v.p = g_cfg.vad;
  return e;
}

std::unique_ptr<AsrClient> MakeAsr() {
  return g_cfg.stt_model.empty() ? MakeWyomingAsrClient() : MakeSherpaAsr(g_cfg.stt_model);
}
std::unique_ptr<TtsClient> MakeTts() {
  return g_cfg.tts_model.empty() ? MakeWyomingTtsClient() : MakeSherpaTts(g_cfg.tts_model);
}
std::unique_ptr<AsrClient> MakeWyomingAsrClient() { return std::make_unique<WyomingAsr>(); }
std::unique_ptr<TtsClient> MakeWyomingTtsClient() { return std::make_unique<WyomingTts>(); }
std::unique_ptr<LlmClient> MakeOllamaClient() { return std::make_unique<Ollama>(); }

void Warmup() {
  auto t0 = std::chrono::steady_clock::now();
  auto ms = [&] { return std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - t0).count(); };
  g_greeting_pcm = Varispeed(MakeTts()->Synthesize(g_cfg.greeting), g_cfg.tts_varispeed);
  Normalize(g_greeting_pcm, g_cfg.tts_rms);
  LOG(INFO) << "[warmup] tts " << ms() << " ms";
  MakeAsr()->Transcribe(std::vector<int16_t>(kAudioRate / 2));
  LOG(INFO) << "[warmup] stt " << ms() << " ms";
  // Loads the model and caches the system-prompt prefix (aborts at the first
  // sentence). If Ollama is down or lacks the model, keep at it in the
  // background, pulling on 404, so SIP still opens and calls get the fallback.
  auto warm = [] {
    return MakeOllamaClient()->Chat({{{"role", "system"}, {"content", SystemPrompt()}},
                                     {{"role", "user"}, {"content", "Hi."}}},
                                    [](auto&) { return false; }).status;
  };
  if (int st = warm(); st == 200) LOG(INFO) << "[warmup] llm " << ms() << " ms";
  else std::thread([st, warm]() mutable {
    for (int wait = 2; st != 200; st = warm(), wait = std::min(wait * 2, 60)) {
      std::this_thread::sleep_for(std::chrono::seconds(wait));
      if (st != 404) continue;
      LOG(WARNING) << "[warmup] pulling " << g_cfg.model << " (may take minutes)";
      httplib::Client cli(g_cfg.ollama);
      cli.set_read_timeout(3600);
      cli.Post("/api/pull", json{{"model", g_cfg.model}, {"stream", false}}.dump(), "application/json");
    }
    LOG(INFO) << "[warmup] llm ready";
  }).detach();
}

// ── Audio cues ────────────────────────────────────────────────────────────────
// A major throughout, one soft bell timbre, levels in dB relative to speech.

struct Note { int semi; float at_ms, ms; };  // semitones from A4 (440 Hz)

// Bell: partials 1-3, higher ones dying faster; 3 ms attack, raised-cosine tail
// to exactly zero (no clicks). Notes may overlap and ring into each other.
static std::vector<int16_t> Bell(std::initializer_list<Note> notes, float db) {
  auto at = [](float ms) { return size_t(ms * kAudioRate / 1000); };
  size_t n = 0;
  for (auto& x : notes) n = std::max(n, at(x.at_ms + x.ms));
  std::vector<float> mix(n);
  for (auto& x : notes) {
    float hz = 440 * std::pow(2.f, x.semi / 12.f), T = x.ms / 1000;
    size_t len = at(x.ms), tail = std::min(len, at(10)), a = at(3);
    for (size_t i = 0; i < len; i++) {
      float t = float(i) / kAudioRate, s = 0;
      for (int k = 1; k <= 3; k++) s += std::pow(.3f, k - 1) * std::exp(-4 * k * t / T) * std::sin(2 * M_PI * k * hz * t);
      float env = std::min(1.f, float(i) / a);
      if (i + tail >= len) env *= .5f * (1 + std::cos(M_PI * (i + tail - len + 1) / tail));
      mix[at(x.at_ms) + i] += s * env;
    }
  }
  double e = 0;
  for (float x : mix) e += x * x;
  float g = g_cfg.tts_rms * std::pow(10.f, (g_cfg.cue_db + db) / 20) / std::sqrt(e / n);
  std::vector<int16_t> v(n);
  for (size_t i = 0; i < n; i++) v[i] = int16_t(std::clamp(mix[i] * g, -29000.f, 29000.f));
  return v;
}

const std::map<std::string, std::vector<int16_t>>& Cues() {
  static const auto c = [] {
    std::map<std::string, std::vector<int16_t>> m = {
        {"answer", Bell({{0, 0, 400}, {4, 90, 400}, {7, 180, 400}, {12, 270, 500}}, 0)},
        {"heard", Bell({{7, 0, 140}, {0, 70, 220}}, -9)},
        {"listening", Bell({{12, 0, 160}, {16, 70, 220}}, -6)},
        {"transfer", Bell({{12, 0, 400}, {7, 100, 400}, {4, 200, 400}, {0, 300, 550}}, -3)},
    };
    // Thinking: one soft arpeggio note per 600 ms, note last, so a short gap
    // stays silent; the loop restarts each gap.
    for (int semi : {-12, -8, -5, -8}) {
      auto note = Bell({{semi, 0, 180}}, -14);
      auto& t = m["thinking"];
      t.resize(t.size() + kAudioRate * 600 / 1000 - note.size());
      t.insert(t.end(), note.begin(), note.end());
    }
    return m;
  }();
  return c;
}

// ── Session ───────────────────────────────────────────────────────────────────

Session::Session(std::string caller, TransferFn transfer, std::unique_ptr<AsrClient> asr,
                 std::unique_ptr<TtsClient> tts, std::unique_ptr<LlmClient> llm)
    : caller_(std::move(caller)), transfer_(std::move(transfer)), asr_(std::move(asr)),
      tts_(std::move(tts)), llm_(std::move(llm)) {}

void Session::Stop() {
  active = false;
  cv_.notify_all();
  llm_->Cancel();
}

Session::~Session() {
  Stop();
  if (thread_.joinable()) thread_.join();
}

void Session::Start() {
  Cue("answer");
  if (!g_greeting_pcm.empty()) {
    audio_out.Push(g_greeting_pcm);
    history_.push_back({{"role", "assistant"}, {"content", g_cfg.greeting}});
  }
  Cue("listening");
  thread_ = std::thread([this] { Run(); });
}

void Session::ProcessRxFrame(const int16_t* s, int n) {
  std::vector<int16_t> utt;
  if (audio_out.Busy()) return vad_->Reset();  // half-duplex: deaf while speaking/thinking
  if (!vad_->Push(s, n, utt)) return;
  { std::lock_guard l(mu_); pending_.push(std::move(utt)); }
  cv_.notify_one();
}

void Session::Cue(const std::string& name) {
  if (g_cfg.cues) audio_out.Push(Cues().at(name));
}

void Session::Thinking(bool on) { audio_out.SetFiller(on && g_cfg.cues ? Cues().at("thinking") : std::vector<int16_t>{}); }

void Session::Say(const std::string& text) {
  LOG(INFO) << caller_ << " [tts] " << text;
  auto pcm = Varispeed(tts_->Synthesize(text), g_cfg.tts_varispeed);
  Normalize(pcm, g_cfg.tts_rms);
  audio_out.Push(pcm);
}

json Session::Messages() const {
  json m = {{{"role", "system"}, {"content", SystemPrompt()}}};
  size_t keep = 2 * g_cfg.history_turns, n = history_.size();
  for (size_t i = n > keep ? n - keep : 0; i < n; i++) m.push_back(history_[i]);
  return m;
}

void Session::Run() {
  while (true) {
    std::vector<int16_t> pcm;
    {
      std::unique_lock l(mu_);
      cv_.wait(l, [&] { return !pending_.empty() || !active; });
      if (!active) return;
      pcm = std::move(pending_.front());
      pending_.pop();
    }
    Cue("heard");
    Thinking(true);
    std::string text = asr_->Transcribe(pcm);
    Thinking(false);
    LOG(INFO) << caller_ << " [stt] " << pcm.size() * 1000 / kAudioRate << " ms → \"" << text << '"';
    if (text.empty()) continue;
    history_.push_back({{"role", "user"}, {"content", text}});

    Thinking(true);  // beeps fill any gap until the reply is fully generated
    bool spoke = false;
    auto r = llm_->Chat(Messages(), [&](const std::string& s) {
      if (!active) return false;
      Say(s), spoke = true;
      return active.load();
    });
    Thinking(false);
    if (!active) return;
    if (!spoke && r.transfer.empty()) Say(r.text = g_cfg.fallback);  // never leave the caller in silence
    history_.push_back({{"role", "assistant"}, {"content", r.text}});

    auto it = g_cfg.directory.find(r.transfer);
    if (it != g_cfg.directory.end() && transfer_) {
      if (auto t = g_cfg.transfer_say; !spoke && !t.empty()) {
        if (auto at = t.find("{}"); at != std::string::npos) t.replace(at, 2, it->first);
        Say(t);
      }
      Cue("transfer");
      while (active && audio_out.Size()) std::this_thread::sleep_for(std::chrono::milliseconds(20));
      LOG(INFO) << caller_ << " [call] transfer " << it->first << " → " << it->second;
      transfer_(it->second);
      active = false;
      return;
    }
    if (!r.transfer.empty()) LOG(WARNING) << caller_ << " [call] unknown department: " << r.transfer;
    Cue("listening");
  }
}

// ── PJSIP glue ────────────────────────────────────────────────────────────────

BotAudioPort::BotAudioPort(Session& s) : s_(s) {
  pj::MediaFormatAudio f;
  f.type = PJMEDIA_TYPE_AUDIO, f.clockRate = kAudioRate, f.channelCount = 1,
  f.bitsPerSample = 16, f.frameTimeUsec = kFrameMs * 1000;
  createPort("bot", f);
}

void BotAudioPort::onFrameRequested(pj::MediaFrame& f) {
  f.type = PJMEDIA_FRAME_TYPE_AUDIO;
  f.buf.resize(f.size);
  s_.audio_out.Pop((int16_t*)f.buf.data(), f.size / 2);
}

void BotAudioPort::onFrameReceived(pj::MediaFrame& f) {
  if (f.type == PJMEDIA_FRAME_TYPE_AUDIO) s_.ProcessRxFrame((const int16_t*)f.buf.data(), f.size / 2);
}

void BotCall::onCallState(pj::OnCallStateParam&) {
  auto ci = getInfo();
  LOG(INFO) << ci.remoteUri << " [call] " << ci.stateText;
  if (ci.state != PJSIP_INV_STATE_DISCONNECTED) return;
  port_.reset();
  if (session_) session_->Stop();
  std::thread([s = std::move(session_)] {}).detach();  // join off the PJSIP thread
  delete this;
}

// Our leg is done once the REFER resolves either way; otherwise it holds a
// call slot (max_calls) until the caller hangs up.
void BotCall::onCallTransferStatus(pj::OnCallTransferStatusParam& p) {
  LOG(INFO) << getInfo().remoteUri << " [call] transfer " << p.statusCode << " " << p.reason;
  if (!p.finalNotify) return;
  p.cont = false;
  pj::CallOpParam op;
  try { hangup(op); } catch (pj::Error& e) { LOG(ERROR) << "[call] hangup: " << e.info(); }
}

void BotCall::onCallMediaState(pj::OnCallMediaStateParam&) {
  auto ci = getInfo();
  for (unsigned i = 0; i < ci.media.size(); i++) {
    if (ci.media[i].type != PJMEDIA_TYPE_AUDIO) continue;
    if (ci.media[i].status == PJSUA_CALL_MEDIA_NONE && session_) session_->Stop();
    auto* aud = ci.media[i].status == PJSUA_CALL_MEDIA_ACTIVE ? dynamic_cast<pj::AudioMedia*>(getMedia(i)) : nullptr;
    if (!aud) continue;
    if (!session_) {
      auto xfer = [cid = getId()](const std::string& ext) {
        static thread_local pj_thread_desc desc;
        static thread_local pj_thread_t* thr;
        if (!thr) pj_thread_register("xfer", desc, &thr);
        std::string uri = "sip:" + ext + (g_cfg.pbx.empty() ? "" : "@" + g_cfg.pbx);
        pj_str_t to = pj_str(uri.data());
        if (pj_status_t st = pjsua_call_xfer(cid, &to, nullptr); st != PJ_SUCCESS) {
          char e[80];
          LOG(ERROR) << "[call] transfer failed: " << pj_strerror(st, e, sizeof e).ptr;
        }
      };
      session_ = std::make_unique<Session>(ci.remoteUri, xfer, MakeAsr(),
                                           MakeTts(), MakeOllamaClient());
      session_->Start();
      port_ = std::make_unique<BotAudioPort>(*session_);
    }
    try {
      aud->startTransmit(*port_), port_->startTransmit(*aud);
      LOG(INFO) << ci.remoteUri << " [media] RTP connected";
    } catch (pj::Error& e) { LOG(ERROR) << "[media] " << e.info(); }
  }
}

void BotAccount::onIncomingCall(pj::OnIncomingCallParam& prm) {
  LOG(INFO) << "[call] incoming from " << prm.rdata.srcAddress;
  auto* c = new BotCall(*this, prm.callId);
  pj::CallOpParam op;
  try {
    if (BotCall::live > g_cfg.max_calls) {
      LOG(WARNING) << c->getInfo().remoteUri << " [call] busy: " << g_cfg.max_calls
                   << " calls up (max_calls); 486";
      op.statusCode = PJSIP_SC_BUSY_HERE;
      return c->hangup(op);
    }
    op.statusCode = PJSIP_SC_OK;
    c->answer(op);
  } catch (pj::Error& e) { LOG(ERROR) << "[call] answer: " << e.info(); }
}
