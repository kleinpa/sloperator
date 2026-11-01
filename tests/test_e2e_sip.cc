// test_e2e_sip.cc — integration tests.
//
// Pipeline: Session with fake clients (no network, no SIP).
// Sip:      the real binary, fake Wyoming/Ollama servers on localhost, and a
//           PJSIP caller that speaks over G.722 RTP and listens for the reply.

#include <arpa/inet.h>
#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>

#include "bot_core.h"

using json = nlohmann::json;
using namespace std::chrono_literals;

static std::vector<int16_t> Sine(int n, double amp = 8000) {
  std::vector<int16_t> v(n);
  for (int i = 0; i < n; i++) v[i] = int16_t(amp * std::sin(2 * M_PI * 440 * i / kAudioRate));
  return v;
}
static bool WaitFor(std::function<bool()> f, std::chrono::milliseconds t = 5s) {
  for (auto end = std::chrono::steady_clock::now() + t; std::chrono::steady_clock::now() < end;
       std::this_thread::sleep_for(10ms))
    if (f()) return true;
  return f();
}

// ═════════════════════════════════════════════════════════════════════════════
// Pipeline
// ═════════════════════════════════════════════════════════════════════════════

struct FakeAsr : AsrClient {
  std::string reply = "hello bot";
  std::atomic_int calls{0};
  std::string Transcribe(const std::vector<int16_t>&) override { return calls++, reply; }
};
struct FakeTts : TtsClient {
  std::vector<std::string> said;
  std::mutex mu;
  std::vector<int16_t> Synthesize(const std::string& t) override {
    std::lock_guard l(mu);
    return said.push_back(t), Sine(kAudioRate / 5, 1000);
  }
};
struct FakeLlm : LlmClient {
  LlmReply reply{"Hello there.", ""};
  std::vector<json> seen;  // messages per call
  std::atomic_int calls{0};
  LlmReply Chat(const json& m, std::function<bool(const std::string&)> say) override {
    seen.push_back(m);
    if (!reply.text.empty()) say(reply.text);
    return calls++, reply;
  }
};

class PipelineTest : public ::testing::Test {
 protected:
  FakeAsr* asr = new FakeAsr;
  FakeTts* tts = new FakeTts;
  FakeLlm* llm = new FakeLlm;
  std::vector<std::string> transfers;
  std::unique_ptr<Session> s;

  void SetUp() override {
    g_cfg = {};
    g_cfg.vad = {500, 200, 100};
    g_cfg.directory = {{"sales", "1042"}};
    g_greeting_pcm = Sine(1600);
  }
  void Start() {
    s = std::make_unique<Session>("sip:t@x", [&](auto& e) { transfers.push_back(e); },
                                  std::unique_ptr<AsrClient>(asr), std::unique_ptr<TtsClient>(tts),
                                  std::unique_ptr<LlmClient>(llm));
    s->Start();
  }
  // Speaks one utterance, draining output like the RTP port would.
  void Speak() {
    auto v = Sine(kFrameSamples), q = std::vector<int16_t>(kFrameSamples);
    for (int i = 0; i < 16; i++) s->audio_out.Clear(), s->ProcessRxFrame((i < 6 ? v : q).data(), kFrameSamples);
  }
  bool Drain(std::function<bool()> until) {
    return WaitFor([&] { return s->audio_out.Clear(), until(); });
  }
};

TEST_F(PipelineTest, RoundTripWithGreetingInHistory) {
  Start();
  EXPECT_GT(s->audio_out.Size(), g_greeting_pcm.size());  // chime + greeting + chime
  Speak();
  ASSERT_TRUE(WaitFor([&] { return llm->calls == 1; }));
  WaitFor([&] { return !tts->said.empty(); });
  EXPECT_EQ(tts->said, std::vector<std::string>{"Hello there."});
  auto& m = llm->seen[0];
  ASSERT_EQ(m.size(), 3u);
  EXPECT_EQ(m[0]["role"], "system");
  EXPECT_EQ(m[1]["content"], g_cfg.greeting);
  EXPECT_EQ(m[2], (json{{"role", "user"}, {"content", "hello bot"}}));
}

TEST_F(PipelineTest, EmptyTranscriptSkipsLlm) {
  asr->reply = "";
  Start();
  Speak();
  ASSERT_TRUE(WaitFor([&] { return asr->calls == 1; }));
  std::this_thread::sleep_for(100ms);
  EXPECT_EQ(llm->calls, 0);
}

TEST_F(PipelineTest, DeafWhileSpeaking) {
  Start();  // greeting still queued
  auto v = Sine(kFrameSamples), q = std::vector<int16_t>(kFrameSamples);
  for (int i = 0; i < 20; i++) s->ProcessRxFrame(v.data(), v.size());
  s->audio_out.Clear();
  for (int i = 0; i < 10; i++) s->ProcessRxFrame(q.data(), q.size());
  std::this_thread::sleep_for(200ms);
  EXPECT_EQ(asr->calls, 0);
}

TEST_F(PipelineTest, TransferToolRoutesViaDirectory) {
  llm->reply = {"", "sales"};  // tool call, no text
  Start();
  Speak();
  ASSERT_TRUE(Drain([&] { return !transfers.empty(); }));
  EXPECT_EQ(transfers, std::vector<std::string>{"1042"});
  EXPECT_EQ(tts->said, std::vector<std::string>{"Connecting you to sales now."});
  EXPECT_TRUE(WaitFor([&] { return !s->active; }));
}

TEST_F(PipelineTest, UnknownDepartmentKeepsListening) {
  llm->reply = {"Let me see.", "accounting"};
  Start();
  Speak();
  ASSERT_TRUE(WaitFor([&] { return llm->calls == 1; }));
  Speak();  // still taking turns
  ASSERT_TRUE(WaitFor([&] { return llm->calls == 2; }));
  EXPECT_TRUE(transfers.empty());
}

TEST_F(PipelineTest, LlmFailureDoesNotLeaveBeepsQueued) {
  // Regression: 30 s of thinking beeps stayed queued, deafening the bot.
  llm->reply = {"", ""};
  Start();
  Speak();
  ASSERT_TRUE(WaitFor([&] { return llm->calls == 1; }));
  ASSERT_TRUE(WaitFor([&] { return s->audio_out.Size() < size_t(kAudioRate); }));
  Speak();
  EXPECT_TRUE(WaitFor([&] { return llm->calls == 2; }));
}

TEST_F(PipelineTest, HistoryIsCapped) {
  g_cfg.history_turns = 2;
  Start();
  for (int i = 1; i <= 4; i++) {
    Speak();
    ASSERT_TRUE(WaitFor([&] { return llm->calls == i; }));
  }
  EXPECT_EQ(llm->seen.back().size(), 1u + 4);  // system + last 2 turns
  EXPECT_EQ(llm->seen.back().back()["role"], "user");
}

TEST(Config, LoadsTypesDirectoryAndPrompt) {
  g_cfg = {};
  LoadConfig(R"(
model: m
think: false
knowledge: Open 9 to 5.
directory: {Peter: 2111, Clock: "0042"}
vad: {threshold: 120}
options: {num_ctx: 2048, temperature: 0.5, stop: "x"}
)");
  EXPECT_EQ(g_cfg.model, "m");
  EXPECT_EQ(g_cfg.think, false);
  EXPECT_EQ(g_cfg.directory.at("Clock"), "0042");
  EXPECT_EQ(g_cfg.vad.threshold, 120);
  EXPECT_EQ(g_cfg.vad.silence_ms, VadParams{}.silence_ms);
  EXPECT_TRUE(g_cfg.options["num_ctx"].is_number_integer());  // Ollama rejects 2048.0
  EXPECT_EQ(g_cfg.options["temperature"], 0.5);
  EXPECT_EQ(g_cfg.options["top_k"], 20);  // defaults kept
  auto p = SystemPrompt();
  EXPECT_NE(p.find("Open 9 to 5."), std::string::npos);
  EXPECT_NE(p.find("Departments: Clock, Peter."), std::string::npos);
}

// ═════════════════════════════════════════════════════════════════════════════
// Sip — real binary over loopback
// ═════════════════════════════════════════════════════════════════════════════

static int Listen(int port) {
  int fd = socket(AF_INET, SOCK_STREAM, 0), one = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  sockaddr_in a{AF_INET, htons(port), {htonl(INADDR_LOOPBACK)}, {}};
  return bind(fd, (sockaddr*)&a, sizeof a) || listen(fd, 8) ? close(fd), -1 : fd;
}
static bool Rx(int fd, void* p, size_t n) {
  for (auto* b = (char*)p; n;) {
    ssize_t r = recv(fd, b, n, 0);
    if (r <= 0) return false;
    b += r, n -= r;
  }
  return true;
}
static void Tx(int fd, const std::string& s) { send(fd, s.data(), s.size(), MSG_NOSIGNAL); }
static std::string Line(int fd) {
  std::string l;
  for (char c; recv(fd, &c, 1, 0) == 1 && c != '\n';) l += c;
  return l;
}
static std::string WyRecv(int fd) {
  json h = json::parse(Line(fd), nullptr, false);
  if (h.is_discarded()) return {};
  std::string skip(h.value("data_length", 0) + h.value("payload_length", 0), 0);
  return Rx(fd, skip.data(), skip.size()) ? h.value("type", "") : "";
}
static void WySend(int fd, const std::string& type, const json& d, const std::string& payload = "") {
  std::string ds = d.dump();
  json h = {{"type", type}, {"data_length", ds.size()}};
  if (!payload.empty()) h["payload_length"] = payload.size();
  Tx(fd, h.dump() + "\n" + ds + payload);
}

// Accept loop; `serve` handles one connection. Counts connections served.
struct FakeServer {
  int fd = -1;
  std::atomic_int hits{0};
  std::thread t;
  bool Start(int port, std::function<bool(int)> serve) {
    if ((fd = Listen(port)) < 0) return false;
    t = std::thread([this, serve] {
      for (int c; (c = accept(fd, nullptr, nullptr)) >= 0; close(c)) hits += serve(c);
    });
    return true;
  }
  ~FakeServer() { shutdown(fd, SHUT_RDWR), close(fd); if (t.joinable()) t.join(); }
};

static bool ServeWhisper(int c) {
  for (std::string t; !(t = WyRecv(c)).empty();)
    if (t == "audio-stop") return WySend(c, "transcript", {{"text", "hello"}}), true;
  return false;
}
static bool ServePiper(int c) {
  if (WyRecv(c) != "synthesize") return false;
  auto pcm = Sine(4000);  // 250 ms
  WySend(c, "audio-start", {{"rate", kAudioRate}, {"width", 2}, {"channels", 1}});
  WySend(c, "audio-chunk", {{"rate", kAudioRate}}, std::string((char*)pcm.data(), pcm.size() * 2));
  WySend(c, "audio-stop", json::object());
  return true;
}
static bool ServeOllama(int c) {
  size_t len = 0;
  for (std::string l; !(l = Line(c)).empty() && l != "\r";)
    if (strncasecmp(l.c_str(), "content-length:", 15) == 0) len = std::stoul(l.substr(15));
  std::string body(len, 0);
  Rx(c, body.data(), len);
  std::string chunk = R"({"message":{"role":"assistant","content":"Hi."},"done":true})" "\n";
  char hex[16];
  snprintf(hex, sizeof hex, "%zx\r\n", chunk.size());
  Tx(c, "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nTransfer-Encoding: chunked\r\n\r\n" +
            std::string(hex) + chunk + "\r\n0\r\n\r\n");
  return json::parse(body, nullptr, false).value("keep_alive", 0) == -1;
}

// Caller's mic/speaker: after `quiet_frames`, a 0.5 s tone then silence;
// records the loudest frame heard back after the tone.
struct CallerPort : pj::AudioMediaPort {
  std::atomic_int frame{0}, quiet_frames{150};
  std::atomic<double> heard{0};
  CallerPort() {
    pj::MediaFormatAudio f;
    f.type = PJMEDIA_TYPE_AUDIO, f.clockRate = kAudioRate, f.channelCount = 1,
    f.bitsPerSample = 16, f.frameTimeUsec = kFrameMs * 1000;
    createPort("caller", f);
  }
  void onFrameRequested(pj::MediaFrame& f) override {
    f.type = PJMEDIA_FRAME_TYPE_AUDIO;
    f.buf.assign(f.size, 0);
    if (int i = frame++ - quiet_frames; i >= 0 && i < 25) {
      auto v = Sine(f.size / 2);
      memcpy(f.buf.data(), v.data(), f.size);
    }
  }
  void onFrameReceived(pj::MediaFrame& f) override {
    if (frame > quiet_frames + 25 && f.size)
      heard = std::max(heard.load(), RmsEnergy((int16_t*)f.buf.data(), f.size / 2));
  }
};

struct CallerCall : pj::Call {
  CallerPort* port;
  std::atomic_bool up{false};
  CallerCall(pj::Account& a, CallerPort* p) : pj::Call(a), port(p) {}
  void onCallState(pj::OnCallStateParam&) override { up = getInfo().state == PJSIP_INV_STATE_CONFIRMED; }
  void onCallMediaState(pj::OnCallMediaStateParam&) override {
    try { auto aud = getAudioMedia(-1); aud.startTransmit(*port), port->startTransmit(aud); }
    catch (pj::Error&) {}
  }
};
struct CallerAccount : pj::Account {};

class SipTest : public ::testing::Test {
 protected:
  static constexpr int kSip = 15060, kOllama = 19434, kPiper = 19200, kWhisper = 19300;
  FakeServer ollama, piper, whisper;
  pid_t pid = -1;
  pj::Endpoint ep;
  std::unique_ptr<CallerAccount> acc;

  void SetUp() override {
    std::string bin = getenv("TEST_SRCDIR") ? std::string(getenv("TEST_SRCDIR")) + "/_main/sloperator" : "";
    if (const char* e = getenv("SLOPERATOR_BIN")) bin = e;
    if (access(bin.c_str(), X_OK)) GTEST_SKIP() << "no sloperator binary (set SLOPERATOR_BIN)";
    ASSERT_TRUE(ollama.Start(kOllama, ServeOllama) && piper.Start(kPiper, ServePiper) &&
                whisper.Start(kWhisper, ServeWhisper));

    char cfg[] = "/tmp/sloperator_test_XXXXXX";
    int fd = mkstemp(cfg);
    std::string y = "max_calls: 2\nvad: {threshold: 300, silence_ms: 300, min_speech_ms: 100}\n";
    ASSERT_EQ(write(fd, y.data(), y.size()), ssize_t(y.size()));
    close(fd);
    if ((pid = fork()) == 0) {
      int null = open("/dev/null", O_WRONLY);
      dup2(null, 1), dup2(null, 2);
      auto at = [](int p) { return "127.0.0.1:" + std::to_string(p); };
      execl(bin.c_str(), bin.c_str(), "--port", std::to_string(kSip).c_str(), "--config_file", cfg,
            "--ollama", ("http://" + at(kOllama)).c_str(), "--piper", at(kPiper).c_str(),
            "--whisper", at(kWhisper).c_str(), nullptr);
      _exit(127);
    }

    ep.libCreate();
    pj::EpConfig ec;
    ec.logConfig.level = ec.logConfig.consoleLevel = 0;
    ec.medConfig.clockRate = ec.medConfig.sndClockRate = kAudioRate;
    ec.medConfig.noVad = true, ec.medConfig.ecTailLen = 0;
    ep.libInit(ec);
    pj::TransportConfig tc;
    tc.port = kSip + 1;
    ep.transportCreate(PJSIP_TRANSPORT_UDP, tc);
    ep.libStart();
    ep.audDevManager().setNullDev();
    ep.codecSetPriority("G722/16000", 255);
    pj::AccountConfig ac;
    ac.idUri = "sip:caller@127.0.0.1";
    acc = std::make_unique<CallerAccount>();
    acc->create(ac);
  }
  void TearDown() override {
    if (acc) ep.hangupAllCalls(), Pump(500ms), acc.reset(), ep.libDestroy();
    if (pid > 0) kill(pid, SIGTERM), waitpid(pid, nullptr, 0);
  }
  bool Pump(std::chrono::milliseconds t, std::function<bool()> done = [] { return false; }) {
    for (auto end = std::chrono::steady_clock::now() + t; std::chrono::steady_clock::now() < end;)
      if (ep.libHandleEvents(20), done()) return true;
    return false;
  }
  CallerCall* Dial(CallerPort& port) {
    auto* c = new CallerCall(*acc, &port);
    pj::CallOpParam op(true);
    op.opt.audioCount = 1, op.opt.videoCount = 0;
    c->makeCall("sip:bot@127.0.0.1:" + std::to_string(kSip), op);
    return c;
  }
};

TEST_F(SipTest, WarmsUpThenAnswersSpeechWithSpeech) {
  // The bot warms every service before opening SIP: greeting TTS, STT, LLM.
  ASSERT_TRUE(Pump(15s, [&] { return piper.hits && whisper.hits && ollama.hits; }));
  CallerPort port;
  CallerCall* call = nullptr;
  // SIP may open a moment after warmup; retry the INVITE until answered.
  for (int i = 0; i < 20 && !(call && call->up); i++) {
    std::this_thread::sleep_for(100ms);
    call = Dial(port);
    Pump(1s, [&] { return call->up.load(); });
  }
  ASSERT_TRUE(call && call->up) << "bot did not answer";
  port.frame = 0;
  // Tone → VAD → STT → LLM (keep_alive sent) → TTS → RTP back to us.
  EXPECT_TRUE(Pump(10s, [&] { return whisper.hits >= 2 && ollama.hits >= 2 && piper.hits >= 2; }))
      << "stt=" << whisper.hits << " llm=" << ollama.hits << " tts=" << piper.hits;
  EXPECT_TRUE(Pump(3s, [&] { return port.heard > 300; })) << "heard rms " << port.heard;
}
