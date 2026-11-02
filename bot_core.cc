// bot_core.cc — Wyoming STT/TTS + Ollama clients, Session pipeline, PJSIP glue.

#include "bot_core.h"

#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
#include <yaml-cpp/yaml.h>

#include <chrono>

#include "absl/log/log.h"

#define CPPHTTPLIB_NO_EXCEPTIONS
#include "httplib.h"

using json = nlohmann::json;

Config g_cfg;
std::vector<int16_t> g_greeting_pcm;

// ── Config ────────────────────────────────────────────────────────────────────

static json Scalar(const YAML::Node& n) {
  try { return n.as<long>(); } catch (...) {}
  try { return n.as<double>(); } catch (...) {}
  try { return n.as<bool>(); } catch (...) {}
  return n.as<std::string>();
}

void LoadConfig(const std::string& text) {
  YAML::Node y = YAML::Load(text);
  auto get = [&](const YAML::Node& n, const char* k, auto& v) {
    if (n[k]) v = n[k].as<std::decay_t<decltype(v)>>();
  };
  auto& c = g_cfg;
  for (auto [k, v] : std::initializer_list<std::pair<const char*, std::string*>>{{"whisper", &c.whisper}, {"ollama", &c.ollama}, {"piper", &c.piper},
                      {"pbx", &c.pbx}, {"model", &c.model}, {"stt_model", &c.stt_model}, {"tts_model", &c.tts_model}, {"language", &c.language},
                      {"greeting", &c.greeting}, {"system_prompt", &c.prompt}, {"knowledge", &c.knowledge}})
    get(y, k, *v);
  get(y, "max_calls", c.max_calls), get(y, "history_turns", c.history_turns), get(y, "threads", c.threads);
  get(y, "tts_rms", c.tts_rms), get(y, "directory", c.directory);
  get(y, "tts_speed", c.tts_speed), get(y, "tts_speaker", c.tts_speaker);
  if (y["think"]) c.think = Scalar(y["think"]);
  if (auto v = y["vad"]) get(v, "model", c.vad.model), get(v, "prob", c.vad.prob),
                         get(v, "threshold", c.vad.threshold), get(v, "silence_ms", c.vad.silence_ms),
                         get(v, "min_speech_ms", c.vad.min_speech_ms);
  if (auto o = y["options"]) for (auto kv : o) c.options[kv.first.as<std::string>()] = Scalar(kv.second);
}

std::string SystemPrompt() {
  std::string p = g_cfg.prompt;
  if (!g_cfg.knowledge.empty())
    p += "\n\nFacts (use only these; never invent details):\n" + g_cfg.knowledge;
  if (!g_cfg.directory.empty()) {
    p += "\n\nTo connect the caller to a department, call transfer_call. Departments:";
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
  timeval tv{30, 0};  // a hung service must not hang the call
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
  LlmReply Chat(const json& messages, std::function<bool(const std::string&)> say) override {
    json req = {{"model", g_cfg.model}, {"messages", messages}, {"options", g_cfg.options},
                {"stream", true}, {"keep_alive", -1}};
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
    cli.set_connection_timeout(5), cli.set_read_timeout(120);
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
        for (auto& s : SplitSentences(buf, first)) emit(s);
      }
      return go;
    };
    httplib::Response res;
    httplib::Error err;
    if (!cli.send(q, res, err) && go) LOG(ERROR) << "[llm] " << httplib::to_string(err);
    else if (res.status && res.status != 200) LOG(ERROR) << "[llm] HTTP " << res.status;
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
  g_greeting_pcm = MakeTts()->Synthesize(g_cfg.greeting);
  Normalize(g_greeting_pcm, g_cfg.tts_rms);
  LOG(INFO) << "[warmup] tts " << ms() << " ms";
  MakeAsr()->Transcribe(std::vector<int16_t>(kAudioRate / 2));
  LOG(INFO) << "[warmup] stt " << ms() << " ms";
  // Loads the model and caches the system-prompt prefix; aborts at first sentence.
  MakeOllamaClient()->Chat({{{"role", "system"}, {"content", SystemPrompt()}},
                            {{"role", "user"}, {"content", "Hi."}}},
                           [](auto&) { return false; });
  LOG(INFO) << "[warmup] llm " << ms() << " ms";
}

// ── Audio cues ────────────────────────────────────────────────────────────────

// Decaying sine burst, repeated every `every_ms` across `total_ms` if given.
static std::vector<int16_t> Tone(float hz, float ms, float decay, float gain,
                                 int total_ms = 0, int every_ms = 0) {
  std::vector<int16_t> v(std::max(total_ms, int(ms)) * kAudioRate / 1000);
  for (size_t s = 0; s < v.size(); s += every_ms ? every_ms * kAudioRate / 1000 : v.size())
    for (int i = 0; i < ms * kAudioRate / 1000 && s + i < v.size(); i++) {
      float t = float(i) / kAudioRate;
      v[s + i] = int16_t(32767 * gain * std::exp(-decay * t) * std::sin(2 * M_PI * hz * t));
    }
  return v;
}
static std::vector<int16_t> AnswerChime() {
  auto a = Tone(659.3f, 120, 6, .35f), b = Tone(880, 280, 6, .35f);
  return a.insert(a.end(), b.begin(), b.end()), a;
}
static auto ListeningChime() { return Tone(1047, 80, 10, .3f); }
static auto SegmentChime() { return Tone(330, 60, 20, .2f); }
static auto ThinkingBeeps(int ms) { return Tone(220, 80, 30, .08f, ms, 600); }

// ── Session ───────────────────────────────────────────────────────────────────

Session::Session(std::string caller, TransferFn transfer, std::unique_ptr<AsrClient> asr,
                 std::unique_ptr<TtsClient> tts, std::unique_ptr<LlmClient> llm)
    : caller_(std::move(caller)), transfer_(std::move(transfer)), asr_(std::move(asr)),
      tts_(std::move(tts)), llm_(std::move(llm)) {}

Session::~Session() {
  active = false;
  cv_.notify_all();
  if (thread_.joinable()) thread_.join();
}

void Session::Start() {
  audio_out.Push(AnswerChime());
  if (!g_greeting_pcm.empty()) {
    audio_out.Push(g_greeting_pcm);
    history_.push_back({{"role", "assistant"}, {"content", g_cfg.greeting}});
  }
  audio_out.Push(ListeningChime());
  thread_ = std::thread([this] { Run(); });
}

void Session::ProcessRxFrame(const int16_t* s, int n) {
  std::vector<int16_t> utt;
  if (audio_out.Size()) return vad_->Reset();  // half-duplex: deaf while speaking
  if (!vad_->Push(s, n, utt)) return;
  { std::lock_guard l(mu_); pending_.push(std::move(utt)); }
  cv_.notify_one();
}

void Session::Say(const std::string& text) {
  LOG(INFO) << caller_ << " [tts] " << text;
  auto pcm = tts_->Synthesize(text);
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
    audio_out.Push(SegmentChime());
    audio_out.Push(ThinkingBeeps(int(pcm.size() * 1000 / kAudioRate) + 2000));
    std::string text = asr_->Transcribe(pcm);
    audio_out.Clear();
    LOG(INFO) << caller_ << " [stt] " << pcm.size() * 1000 / kAudioRate << " ms → \"" << text << '"';
    if (text.empty()) continue;
    history_.push_back({{"role", "user"}, {"content", text}});

    audio_out.Push(ThinkingBeeps(30000));
    bool spoke = false;
    auto r = llm_->Chat(Messages(), [&](const std::string& s) {
      if (!active) return false;
      if (!spoke) audio_out.Clear(), spoke = true;
      Say(s);
      return active.load();
    });
    if (!spoke) audio_out.Clear();  // LLM failed or only called a tool
    history_.push_back({{"role", "assistant"}, {"content", r.text}});
    if (!active) return;

    auto it = g_cfg.directory.find(r.transfer);
    if (it != g_cfg.directory.end() && transfer_) {
      if (!spoke) Say("Connecting you to " + it->first + " now.");
      while (active && audio_out.Size()) std::this_thread::sleep_for(std::chrono::milliseconds(20));
      LOG(INFO) << caller_ << " [call] transfer " << it->first << " → " << it->second;
      transfer_(it->second);
      active = false;
      return;
    }
    if (!r.transfer.empty()) LOG(WARNING) << caller_ << " [call] unknown department: " << r.transfer;
    audio_out.Push(ListeningChime());
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
  if (session_) session_->active = false;
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
    if (ci.media[i].status == PJSUA_CALL_MEDIA_NONE && session_) session_->active = false;
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
  pj::CallOpParam op;
  op.statusCode = PJSIP_SC_OK;
  try { (new BotCall(*this, prm.callId))->answer(op); }
  catch (pj::Error& e) { LOG(ERROR) << "[call] answer: " << e.info(); }
}
