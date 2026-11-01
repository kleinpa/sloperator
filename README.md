# sloperator

SIP voice receptionist: answers G.722 calls, transcribes with faster-whisper,
replies with an Ollama LLM, speaks with Piper, and transfers via a directory.

```
caller ⇄ PJSIP (G.722, 16 kHz) ⇄ VAD → Whisper ─text→ Ollama ─sentences→ Piper ─PCM→ caller
                                                        └─ transfer_call tool → SIP REFER
```

- **Streaming:** Piper starts on the first sentence (or first long clause).
- **Warmup:** every service is warmed before SIP opens, so the first call isn't
  cold and the greeting is ready.
- **Half-duplex:** caller audio is ignored while the bot is talking.

## Services

| Service | Default | Image |
| --- | --- | --- |
| STT | `whisper:10300` (Wyoming) | `rhasspy/wyoming-faster-whisper` |
| LLM | `http://ollama:11434` | `ollama/ollama` |
| TTS | `piper:10200` (Wyoming) | `rhasspy/wyoming-piper` |

On CPU, set these on Ollama: `OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1
OLLAMA_KV_CACHE_TYPE=q8_0`.

## Build / run / test

```sh
bazel build -c opt //:sloperator
bazel run -c opt //:sloperator -- --config_file=$PWD/bot_config.yaml
bazel test -c opt //tests:all   # unit + pipeline + real SIP call vs fake services
```

Flags: `--config_file`, `--port` (5060), `--public_addr`, and `--whisper`,
`--ollama`, `--piper`, `--pbx` to override the config. Put machine-specific Bazel
settings (proxy, registry mirror) in `user.bazelrc`.

## Config

See [bot_config.yaml](bot_config.yaml). Key points:

- `system_prompt` + `knowledge` + the `directory` names make up the system prompt.
- Transfers use Ollama tool calling, so the model needs tool support (Qwen3 has it; Gemma 3 doesn't).
- Ollama loads the model with `keep_alive: -1`, so it is never unloaded.
