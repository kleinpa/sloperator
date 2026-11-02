// main.cc — flags + config, warm up services, then run the PJSIP event loop.

#include <yaml-cpp/yaml.h>

#include <csignal>
#include <fstream>
#include <iostream>
#include <sstream>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "bot_core.h"

ABSL_FLAG(std::string, config_file, "", "YAML config (see bot_config.yaml)");
ABSL_FLAG(std::string, whisper, "", "Wyoming STT host:port (overrides config)");
ABSL_FLAG(std::string, ollama, "", "Ollama URL (overrides config)");
ABSL_FLAG(std::string, piper, "", "Wyoming TTS host:port (overrides config)");
ABSL_FLAG(std::string, pbx, "", "SIP host for transfers (overrides config)");
ABSL_FLAG(int, port, 5060, "SIP UDP listen port");
ABSL_FLAG(std::string, public_addr, "", "Public IP for SDP/Contact (NAT)");
ABSL_FLAG(bool, print_config, false, "Print the effective config (defaults + file + flags) and exit");

static std::atomic_bool g_running{true};

int main(int argc, char* argv[]) {
  for (int s : {SIGINT, SIGTERM}) std::signal(s, [](int) { g_running = false; });
  std::signal(SIGPIPE, SIG_IGN);
  absl::SetProgramUsageMessage("SIP voice bot: Whisper STT + Ollama LLM + Piper TTS.");
  absl::ParseCommandLine(argc, argv);
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  absl::InitializeLog();

  if (auto f = absl::GetFlag(FLAGS_config_file); !f.empty()) {
    std::stringstream ss;
    ss << std::ifstream(f).rdbuf();
    try { for (auto& k : LoadConfig(ss.str())) LOG(WARNING) << "[config] unknown key: " << k; }
    catch (const YAML::Exception& e) { LOG(FATAL) << "[config] " << f << ": " << e.what(); }
  }
  for (auto [flag, v] : std::initializer_list<std::pair<absl::Flag<std::string>*, std::string*>>{
           {&FLAGS_whisper, &g_cfg.whisper}, {&FLAGS_ollama, &g_cfg.ollama},
                         {&FLAGS_piper, &g_cfg.piper}, {&FLAGS_pbx, &g_cfg.pbx}})
    if (auto s = absl::GetFlag(*flag); !s.empty()) *v = s;

  if (absl::GetFlag(FLAGS_print_config)) return std::cout << DumpConfig() << "\n", 0;
  LOG(INFO) << "stt=" << g_cfg.whisper << " llm=" << g_cfg.ollama << " (" << g_cfg.model
            << ") tts=" << g_cfg.piper;
  Warmup();  // before SIP opens: no cold first call, greeting ready

  pj::Endpoint ep;
  try {
    ep.libCreate();
    pj::EpConfig cfg;
    cfg.uaConfig.maxCalls = g_cfg.max_calls;
    cfg.logConfig.level = cfg.logConfig.consoleLevel = 3;
    cfg.medConfig.clockRate = cfg.medConfig.sndClockRate = kAudioRate;
    cfg.medConfig.noVad = true;
    cfg.medConfig.ecTailLen = 0;
    ep.libInit(cfg);

    pj::TransportConfig tc;
    tc.port = absl::GetFlag(FLAGS_port);
    tc.publicAddress = absl::GetFlag(FLAGS_public_addr);
    ep.transportCreate(PJSIP_TRANSPORT_UDP, tc);
    try { ep.transportCreate(PJSIP_TRANSPORT_UDP6, tc); } catch (...) {}
    ep.libStart();
    ep.audDevManager().setNullDev();
    // G.722 only. The codec ID uses its 16 kHz audio clock, not the 8 kHz RTP clock.
    ep.codecSetPriority("G722/16000", 255);
    ep.codecSetPriority("PCMU/8000", 0);
    ep.codecSetPriority("PCMA/8000", 0);

    pj::AccountConfig ac;
    ac.idUri = "sip:bot@0.0.0.0";
    BotAccount account;
    account.create(ac);
    LOG(INFO) << "listening on UDP " << tc.port;
    while (g_running) ep.libHandleEvents(100);
    account.shutdown();
    ep.libDestroy();
  } catch (pj::Error& e) {
    LOG(FATAL) << e.info();
  }
}
