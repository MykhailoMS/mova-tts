# mova-tts

Ukrainian text-to-speech with voice cloning. The model is a fine-tune of
[neuphonic/neutts-air](https://huggingface.co/neuphonic/neutts-air) on ~1780 hours of Ukrainian
speech. It runs through llama.cpp (GGUF) with the [NeuCodec](https://huggingface.co/neuphonic/neucodec)
audio codec. It streams, with the first audio after 0.16 s on an RTX 4090, and it runs faster
than real time on a CPU.

```bash
pip install mova-tts
```

```python
from mova_tts import MovaTTS

tts = MovaTTS()                                        # downloads the model on first use
voice = tts.voice("me.wav", "Точний текст, який звучить у записі.")
wav = tts.synthesize("Доброго дня! Сьогодні я розповім вам про нашу нову систему.", voice)
tts.save(wav, "out.wav")
```

Clone any voice from a clean 3–12 second recording plus the exact text spoken in it. Encode a
voice once and reuse it with `voice.save("me.npz")` and `Voice.load("me.npz")`. There is an
example voice from Common Voice (CC0) in `examples/`:

```bash
mova-tts "Доброго дня! Як ваші справи?" --ref examples/voice_cv.wav \
    --ref-text "$(cat examples/voice_cv.txt)" -o out.wav
```

## Streaming

```python
for chunk in tts.stream("Перше речення. Друге речення.", voice):
    play(chunk)                        # float32, 24 kHz mono
```

Audio comes in one-second chunks. The first chunk is 0.2 s long and needs 35 generated codes.
Long text is generated sentence by sentence.

## Speed

Time to first audio (front-end included) and real-time factor (RTF: generation time divided
by audio length). Median over 6 sentences, measured on one machine:

| hardware | first audio | RTF |
|---|---:|---:|
| NVIDIA RTX 4090 (Q8_0 or Q4_K_M) | 0.16 s | 0.14 |
| Intel i9-13900KF, CPU only, 8 threads (Q8_0 or Q4_K_M) | 1.0 s | 0.72 |

`device="auto"` uses CUDA if it is available, then the Apple GPU (Metal), then the CPU. On the
CPU and on a Mac, the audio is decoded with the NeuCodec ONNX decoder. On the CPU, set
`threads=` to the number of performance cores. More threads than that slow it down.

## Quality

The model was evaluated on 234 held-out sentences (Common Voice and audiobooks, speakers never
seen in training), cloning each speaker from a different sentence of theirs. Each sentence is
sampled, so every row gives the range over 3 seeds (2 for Q4_K_M):

| | WER | CER | UTMOS | speaker similarity |
|---|---:|---:|---:|---:|
| `synthesize()`, Q8_0 | 0.123–0.140 | 0.035–0.041 | 3.45 | 0.48–0.50 |
| `stream()`, Q8_0 | 0.135–0.140 | 0.041–0.046 | 3.37 | 0.48–0.50 |
| `stream()`, Q4_K_M | 0.131–0.135 | 0.041 | 3.33 | 0.48–0.49 |

WER and CER come from Whisper large-v3 transcripts of the generated audio. UTMOS is a
predicted naturalness score from 1 to 5. Speaker similarity is the ECAPA cosine to the
speaker's real recordings. A take that does not stop, or stops too early, is regenerated
with another seed. `stream()` can do that only before its first chunk is out.

## Text front-end

1. Clean-up, then numbers, dates and units turned into words with
   [skypro1111/m2m100-ukr-verbalization](https://huggingface.co/skypro1111/m2m100-ukr-verbalization).
   That model (~0.5 GB) downloads the first time a text contains digits.
2. Latin words transliterated, with acronyms spelled out by letter names: CRM becomes сі-ар-ем.
3. Stress from [naholos](https://github.com/MykhailoMS/naholos).
4. IPA from [ipa-uk](https://github.com/patriotyk/ipa-uk), vendored here under MIT.

If a word is stressed wrong, mark it yourself and your mark always wins. You can write
`за+мок` or `за́мок` in the text, or keep a list of such words in a file:

```python
tts = MovaTTS(user_dict="my_words.txt")     # one stressed word per line: за+мок
print(tts.frontend.stressed("Старий замок"))  # check the stress before synthesis
```

## Options

```python
MovaTTS(quant="Q4_K_M")     # smaller model: 0.6 GB instead of 0.8 GB
MovaTTS(device="cpu")       # "auto" | "cuda" | "metal" | "cpu"
MovaTTS(seed=42)            # repeatable takes; default is a new take every call
MovaTTS(watermark=False)    # synthesize() adds a Perth watermark by default
```

Each take is sampled, so it sounds a little different every time. Sometimes a take stretches
a word or pauses oddly. If that happens, generate again or use another seed.

## Model and training

- Base: [neuphonic/neutts-air](https://huggingface.co/neuphonic/neutts-air), a 0.7B Qwen-based
  speech language model (Apache-2.0). It was fully fine-tuned in two stages: 12k steps on a
  740-hour subset, then 20k steps on all the data.
- Data: ~1780 hours of Ukrainian speech: audiobooks read by professional narrators, podcasts,
  Common Voice, and studio voice recordings. All audio was re-encoded with NeuCodec. Stress
  labels come from the recordings themselves: the text was aligned with the Montreal Forced
  Aligner, and an acoustic detector marked the stress each speaker actually used.
- Weights: [Mikhailo/mova-tts](https://huggingface.co/Mikhailo/mova-tts) holds safetensors (for
  transformers) and GGUF Q8_0 and Q4_K_M files (for llama.cpp).

## Responsible use

Clone only voices you have permission to use. `synthesize()` output carries the imperceptible
[Perth](https://github.com/resemble-ai/Perth) watermark, so generated audio can be detected.

## License

Code and model: Apache-2.0, like NeuTTS-Air and NeuCodec. The vendored ipa-uk is MIT
(`LICENSES/ipa-uk.txt`).
