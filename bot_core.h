#pragma once
// bot_core.h — config, AI client interfaces, per-call Session, PJSIP glue.
//
//   PJSIP ⇄ BotAudioPort ⇄ Session: VAD → AsrClient → LlmClient → TtsClient
//
// Clients are injectable so tests run the full pipeline without network.

#include <pjsua2.hpp>

#include <nlohmann/json.hpp>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <thread>

#include "chatbot_lib.h"

// ── Config (YAML; see bot_config.yaml) ────────────────────────────────────────
struct Config {
  std::string whisper = "whisper:10300", ollama = "http://ollama:11434",
              piper = "piper:10200", pbx;
  std::string model = "qwen3:4b-instruct", language = "en",
              greeting = "Hello, how can I help you?",
              fallback = "Sorry, I'm having trouble right now. Please try again in a moment.",
              prompt = "You are a helpful voice assistant answering a phone call. "
                       "Reply in one to three short, plain spoken sentences: no "
                       "lists, markdown or emoji. If you don't know, say so.",
              knowledge;                          // facts appended to prompt
  std::map<std::string, std::string> directory;  // department → extension
  nlohmann::json options = {{"temperature", 0.3}, {"top_p", 0.8},
                            {"top_k", 20}, {"num_ctx", 4096}};
  nlohmann::json think;  // null = don't send
  std::string stt_model, tts_model;  // sherpa model dirs; empty = Wyoming
  double tts_speed = 1;               // tempo only (pitch kept)
  double tts_varispeed = 1;          // tempo + pitch; 0.726 = old derpy voice
  int tts_speaker = 0;               // multi-voice models (Kokoro)
  int max_calls = 1, history_turns = 8, threads = 4;  // threads: in-process models
  double tts_rms = 4000;  // speech loudness, ~-18 dBFS
  VadParams vad;
};
extern Config g_cfg;
extern std::vector<int16_t> g_greeting_pcm;  // synthesized once by Warmup()

// Parses YAML text into g_cfg (throws YAML::Exception).
void LoadConfig(const std::string& yaml);
// Full system prompt: prompt + knowledge + transfer instructions.
std::string SystemPrompt();
// Loads every model and synthesizes the greeting; call before taking calls.
void Warmup();

// ── Clients ───────────────────────────────────────────────────────────────────
struct AsrClient {
  virtual ~AsrClient() = default;
  virtual std::string Transcribe(const std::vector<int16_t>& pcm) = 0;  // "" on failure
};
struct TtsClient {
  virtual ~TtsClient() = default;
  virtual std::vector<int16_t> Synthesize(const std::string& text) = 0;  // kAudioRate PCM
};
struct LlmReply { std::string text, transfer; int status = 200; };  // transfer = department; status = HTTP (0: unreachable)
struct LlmClient {
  virtual ~LlmClient() = default;
  // Streams sentences to `say`; it returns false to abort.
  virtual LlmReply Chat(const nlohmann::json& messages,
                        std::function<bool(const std::string&)> say) = 0;
  virtual void Cancel() {}  // any thread: abort in-flight and future Chats
};
// Voice activity: feed 20 ms frames; true when `utt` holds a full utterance.
struct VadEngine {
  virtual ~VadEngine() = default;
  virtual bool Push(const int16_t* s, int n, std::vector<int16_t>& utt) = 0;
  virtual void Reset() = 0;
};
std::unique_ptr<VadEngine> MakeVad();  // Silero if g_cfg.vad.model, else energy
std::unique_ptr<VadEngine> MakeSileroVad(const VadParams&);
std::vector<int16_t> ReadWav(const std::string& path);  // any rate → kAudioRate
std::unique_ptr<AsrClient> MakeAsr();  // in-process if g_cfg.stt_model, else Wyoming
std::unique_ptr<AsrClient> MakeSherpaAsr(const std::string& model_dir);
std::unique_ptr<AsrClient> MakeWyomingAsrClient();
std::unique_ptr<TtsClient> MakeTts();  // in-process if g_cfg.tts_model, else Wyoming
std::unique_ptr<TtsClient> MakeSherpaTts(const std::string& model_dir);
std::unique_ptr<TtsClient> MakeWyomingTtsClient();
std::unique_ptr<LlmClient> MakeOllamaClient();

// ── Session: one call's pipeline thread ───────────────────────────────────────
class Session {
 public:
  using TransferFn = std::function<void(const std::string& ext)>;
  Session(std::string caller, TransferFn transfer, std::unique_ptr<AsrClient>,
          std::unique_ptr<TtsClient>, std::unique_ptr<LlmClient>);
  ~Session();
  Session(const Session&) = delete;

  void Start();
  void Stop();  // hangup: ends the pipeline, aborting any LLM request
  void ProcessRxFrame(const int16_t* s, int n);  // PJSIP audio thread

  AudioQueue audio_out;
  std::atomic_bool active{true};

 private:
  void Run();
  void Say(const std::string& text);
  nlohmann::json Messages() const;

  std::string caller_;
  TransferFn transfer_;
  std::unique_ptr<AsrClient> asr_;
  std::unique_ptr<TtsClient> tts_;
  std::unique_ptr<LlmClient> llm_;
  std::unique_ptr<VadEngine> vad_ = MakeVad();
  nlohmann::json history_ = nlohmann::json::array();
  std::thread thread_;
  std::queue<std::vector<int16_t>> pending_;
  std::mutex mu_;
  std::condition_variable cv_;
};

// ── PJSIP glue ────────────────────────────────────────────────────────────────
class BotAudioPort : public pj::AudioMediaPort {
 public:
  explicit BotAudioPort(Session& s);
  void onFrameRequested(pj::MediaFrame& f) override;
  void onFrameReceived(pj::MediaFrame& f) override;

 private:
  Session& s_;
};

class BotCall : public pj::Call {
 public:
  using pj::Call::Call;
  void onCallState(pj::OnCallStateParam&) override;
  void onCallMediaState(pj::OnCallMediaStateParam&) override;
  void onCallTransferStatus(pj::OnCallTransferStatusParam&) override;

 private:
  std::unique_ptr<Session> session_;
  std::unique_ptr<BotAudioPort> port_;
};

struct BotAccount : pj::Account {
  void onIncomingCall(pj::OnIncomingCallParam& prm) override;
};
