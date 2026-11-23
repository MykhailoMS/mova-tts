# mova-tts

Ukrainian text-to-speech with voice cloning. The model is a fine-tune of
[neuphonic/neutts-air](https://huggingface.co/neuphonic/neutts-air) on ~1460 hours of Ukrainian
speech. It runs through llama.cpp (GGUF) with the [NeuCodec](https://huggingface.co/neuphonic/neucodec)
audio codec. It streams, with the first audio after 0.16 s on an RTX 4090, and it runs faster
than real time on a CPU.

```bash
pip install mova-tts
hf auth login        # once: the NeuCodec audio codec is a gated repo, see below
```

The audio codec comes from Neuphonic's gated Hugging Face repos. Open
[neuphonic/neucodec](https://huggingface.co/neuphonic/neucodec) and
[neuphonic/neucodec-onnx-decoder](https://huggingface.co/neuphonic/neucodec-onnx-decoder) once,
accept the terms (access is granted at once), and log in with `hf auth login`. The mova-tts
model itself is public.

```python
from mova_tts import MovaTTS, preset, presets

tts = MovaTTS()                                        # downloads the model on first use
print(presets())                                       # ['lada', 'mykyta', 'oleksa', 'tetiana']
wav = tts.synthesize("Доброго дня! Сьогодні я розповім вам про нашу нову систему.", preset("tetiana"))
tts.save(wav, "out.wav")
```

| preset | voice | UTMOS | WER |
|---|---|---:|---:|
| `tetiana` (default) | female | 3.64 | 0.020 |
| `mykyta` | male | 3.67 | 0.000 |
| `oleksa` | male | 3.41 | 0.020 |
| `lada` | female, noisier source recordings | 2.47 | 0.029 |

The preset voices come from the [OpenTTS](https://huggingface.co/datasets/speech-uk/opentts-tetiana)
datasets (Apache-2.0). The scores are from 8 takes of 4 test sentences per voice. A preset is
already encoded, so it needs no recording. Hear samples on the
[model page](https://huggingface.co/Mikhailo/mova-tts/tree/main/samples).

To clone any voice, see [Your own voice](#your-own-voice) below. Another voice to try cloning
with is `tts.example_voice()`, a Common Voice recording (CC0) bundled as a wav.

```bash
mova-tts "Доброго дня! Як ваші справи?" --preset mykyta -o out.wav
```

## Your own voice

You need two things: a short recording of the voice and the exact text spoken in it.

**1. Record 5–10 seconds.** One speaker, reading one or two ordinary sentences in the tone
you want to hear back. The model copies the timbre, the pace, the mood and the room of the
recording.

- Pick a quiet room, with no music, no echo and no other voices. Keep the mic 15–30 cm away.
- Read calmly and naturally. A cheerful or dialog-like reading carries over into everything
  generated.
- Any format works: wav, flac, ogg, mp3, or m4a from Voice Memos. Leading and trailing silence
  is trimmed automatically, and anything past 12 s is cut off.

**2. Write down exactly what you said**, word for word, including numbers as you said them.
The model learns how the text maps to the sound from this pair. A wrong transcript gives
slurred or skipped words.

**3. Encode the voice once and save it:**

```python
from mova_tts import MovaTTS, Voice

tts = MovaTTS()
voice = tts.voice("me.m4a", "Учора ввечері я довго гуляв парком, бо погода була чудова.")
voice.save("me.npz")                                # ~3 KB: codes + phonemes, no audio

voice = Voice.load("me.npz")                        # later, in any script: no recording needed
tts.save(tts.synthesize("Будь-який текст моїм голосом.", voice), "out.wav")
for chunk in tts.stream("Або потоком, для живої розмови.", voice):
    play(chunk)                                     # float32, 24 kHz mono
```

From the command line:

```bash
mova-tts "Перевірка мого голосу." --ref me.m4a \
    --ref-text "Учора ввечері я довго гуляв парком, бо погода була чудова." \
    --save-voice me.npz -o test.wav
mova-tts "Будь-який текст." --voice me.npz -o out.wav
```

**If it sounds off:**

- Words get skipped or swallowed: check the transcript, and use a longer and clearer clip.
- Noise or echo in the output: the reference has them, so re-record in a quieter spot.
- Odd pauses or stretched words: that is an unlucky take. Generate again, or pass `seed=`.
- The voice is not similar enough: try a different sentence. Usually one of 2–3 recordings
  clones noticeably better.

Encoding a recording downloads the NeuCodec encoder (`neuphonic/neucodec`, gated, see
[install](#mova-tts)) the first time. A saved `.npz` voice is never encoded again.

Clone only your own voice or a voice whose owner agreed to it.

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
seen in training), cloning each speaker from a different sentence of theirs, through this
package (GGUF, the same seed for every row):

| | WER | CER | UTMOS | speaker similarity |
|---|---:|---:|---:|---:|
| `synthesize()`, Q8_0 | 0.114 | 0.032 | 3.53 | 0.49 |
| `stream()`, Q8_0 | 0.119 | 0.033 | 3.44 | 0.49 |
| `stream()`, Q4_K_M | 0.138 | 0.046 | 3.42 | 0.48 |

Version 0.1 of the model, on the same sentences and seed: 0.123 / 0.136 / 0.131 WER and
3.46 / 3.37 / 3.33 UTMOS. Sampling makes a single run vary by about ±0.01 WER.

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
  speech language model (Apache-2.0), fully fine-tuned from the base for 30k steps.
- Data: ~1460 hours of Ukrainian speech: audiobooks read by professional narrators, podcasts,
  Common Voice, and studio voice recordings. Every clip was scored with UTMOS and those below
  2.5 were dropped (~280 h), and no single narrator has more than 20 hours. All audio was
  re-encoded with NeuCodec. Stress labels come from the recordings themselves: the text was
  aligned with the Montreal Forced Aligner, and an acoustic detector marked the stress each
  speaker actually used.
- Weights: [Mikhailo/mova-tts](https://huggingface.co/Mikhailo/mova-tts) holds safetensors (for
  transformers) and GGUF Q8_0 and Q4_K_M files (for llama.cpp).

## Responsible use

Clone only voices you have permission to use. `synthesize()` output carries the imperceptible
[Perth](https://github.com/resemble-ai/Perth) watermark, so generated audio can be detected.

## License

Code and model: Apache-2.0, like NeuTTS-Air and NeuCodec. The vendored ipa-uk is MIT
(`LICENSES/ipa-uk.txt`).
