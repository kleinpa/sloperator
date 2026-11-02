# sloperator

SIP voice receptionist: answers G.722 calls, listens with Silero VAD and
Moonshine, replies with an Ollama LLM, speaks with Piper, and transfers calls
via a directory.

```
caller ⇄ PJSIP (G.722, 16 kHz) ⇄ VAD → STT ─text→ Ollama ─sentences→ TTS ─PCM→ caller
                                │                └─ transfer_call tool → SIP REFER
                    in-process (sherpa-onnx): Silero, Moonshine/Parakeet, Piper/Kokoro
```

- **Two processes:** the bot and Ollama. VAD, STT and TTS run in-process through
  sherpa-onnx (about 350 MB RSS for all three). Wyoming Whisper/Piper services
  still work if you leave out `stt_model`/`tts_model`.
- **Streaming:** TTS starts on the first sentence (or first long clause).
- **Warmup:** every model is loaded before SIP opens, so the first call isn't
  cold and the greeting is ready.
- **Half-duplex:** caller audio is ignored while the bot is talking.

On CPU, set these on Ollama: `OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=1
OLLAMA_KV_CACHE_TYPE=q8_0`.

## Build / run / test

```sh
bazel build -c opt //:sloperator
bazel test  -c opt //tests:all          # unit, pipeline, SIP calls (fake LLM, real models)
bazel test  -c opt //tests:test_sherpa_parakeet //tests:test_sherpa_kokoro   # optional models
bazel run   -c opt //:image_load        # container with models at /models
```

The heaviest test places a real SIP call: the caller *says* "transfer me to
sales" (Piper), and the bot must hear it through Silero and Moonshine and turn
the LLM's tool call into a SIP REFER.

Flags: `--config_file`, `--port` (5060), `--public_addr`, and `--whisper`,
`--ollama`, `--piper`, `--pbx` to override the config. Put machine-specific Bazel
settings (proxy, registry mirror) in `user.bazelrc`.

The image's glibc comes from Debian 12, so build it on a host with glibc ≤ 2.36
(as CI does on ubuntu-22.04).

## Config

Every key has a default, but a real deployment needs at least this
(`--config_file`):

```yaml
ollama: http://ollama:11434          # Ollama endpoint
model: qwen3:4b-instruct             # must support tool calling
stt_model: /models/moonshine         # in-process models (paths in the image)
tts_model: /models/piper
vad: {model: /models/silero_vad.onnx}

greeting: Hello, Example Co, how can I help you?
system_prompt: |
  You are the phone receptionist for Example Co. Reply in one to three short,
  plain spoken sentences.
knowledge: |                         # facts the bot may state; nothing else
  Open 9 to 5, Monday to Friday.
directory:                           # department → extension (transfer_call tool)
  sales: 1042
  support: 1050
pbx: pbx.local                       # SIP host transfers are sent to
```

**Concurrent calls:** set `max_calls: N` and `OLLAMA_NUM_PARALLEL=N`. Each call
gets its own pipeline. In-process STT/TTS models are shared and take turns
(about 0.3 s each). On CPU, parallel LLM token generation is close to free,
because it's memory-bandwidth bound, but prompt processing is not: N callers
speaking at once each wait longer.

Run with `--public_addr <ip>` if the bot is behind NAT. Without `stt_model` or
`tts_model`, it uses the Wyoming services at `whisper:` / `piper:` instead.

See [bot_config.yaml](bot_config.yaml) for every key. Key points:

- `system_prompt` + `knowledge` + the `directory` names make up the system prompt.
- Transfers use Ollama tool calling, so the model needs tool support (Qwen3 has it; Gemma 3 doesn't).
- Ollama loads the model with `keep_alive: -1`, so it is never unloaded.
- sherpa-onnx is pinned to 1.13.8: 2.0 removed espeak-ng, which Piper/Kokoro need for text input.
