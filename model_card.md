---
language:
- uk
license: apache-2.0
base_model: neuphonic/neutts-air
pipeline_tag: text-to-speech
library_name: mova-tts
tags:
- ukrainian
- text-to-speech
- voice-cloning
- neutts
- gguf
- llama.cpp
---

# mova-tts

Ukrainian text-to-speech with voice cloning. This model is a **fine-tune of
[neuphonic/neutts-air](https://huggingface.co/neuphonic/neutts-air)** on ~1780 hours of
Ukrainian speech. It generates [NeuCodec](https://huggingface.co/neuphonic/neucodec) tokens
from IPA phonemes, and it clones a voice from a 3–12 second reference recording.

```bash
pip install mova-tts
```

```python
from mova_tts import MovaTTS

tts = MovaTTS()                      # GGUF Q8_0 from this repo; CUDA, Apple GPU or CPU
voice = tts.voice("me.wav", "Точний текст, який звучить у записі.")
tts.save(tts.synthesize("Доброго дня! Як ваші справи?", voice), "out.wav")

for chunk in tts.stream("Перше речення. Друге речення.", voice):
    ...                              # float32 24 kHz chunks, the first one after 35 codes
```

The [mova-tts](https://github.com/MykhailoMS/mova-tts) package does the Ukrainian text
front-end: numbers to words, stress with [naholos](https://github.com/MykhailoMS/naholos), and
IPA. It also handles streaming decoding, with left context and a crossfade.

## Files

| file | for |
|---|---|
| `model.safetensors`, `config.json`, tokenizer files | transformers (bf16, 0.7B) |
| `gguf/mova-tts-Q8_0.gguf` (0.8 GB) | llama.cpp, the default |
| `gguf/mova-tts-Q4_K_M.gguf` (0.6 GB) | llama.cpp, smaller |

The prompt format is the same as NeuTTS-Air with phoneme input:
`user: Convert the text to speech:<|TEXT_PROMPT_START|>{ref IPA} {IPA}<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{ref codes}`.
The IPA must come from the mova-tts front-end (ipa-uk with stress marks). eSpeak phonemes
will not work.

## Evaluation

The evaluation used 234 held-out sentences from Common Voice and audiobooks, with speakers
that were never seen in training. Each speaker was cloned from a different sentence of theirs:

| | WER | CER | UTMOS | speaker similarity |
|---|---:|---:|---:|---:|
| whole utterance, Q8_0 | 0.123–0.140 | 0.035–0.041 | 3.45 | 0.48–0.50 |
| streaming, Q8_0 | 0.135–0.140 | 0.041–0.046 | 3.37 | 0.48–0.50 |
| streaming, Q4_K_M | 0.131–0.135 | 0.041 | 3.33 | 0.48–0.49 |

Each row is the range over 3 sampling seeds (2 for Q4_K_M).

WER and CER come from Whisper large-v3 transcripts. UTMOS is a predicted naturalness score from 1 to 5.
Speaker similarity is the ECAPA cosine to the speaker's real recordings.

Speed (streaming, front-end included): first audio after 0.16 s on an RTX 4090 (RTF 0.14),
after 1.0 s on an i9-13900KF CPU with 8 threads (RTF 0.72).

## Training

- **Base:** [neuphonic/neutts-air](https://huggingface.co/neuphonic/neutts-air), a 0.7B
  Qwen-based speech LM (Apache-2.0). Full fine-tune in two stages: 12k steps on a 740-hour
  subset, then 20k steps on all the data (LR 2e-5).
- **Data:** ~1780 hours of Ukrainian speech: audiobooks read by professional narrators,
  podcasts, Common Voice, and studio voice recordings. Leading and trailing silence was trimmed,
  utterances with long internal pauses were dropped, and all audio was re-encoded with
  NeuCodec. The training data is not released.
- **Text:** IPA from ipa-uk. The stress labels were taken from the audio itself: the text was
  aligned with the Montreal Forced Aligner, and an acoustic detector marked the stress the
  speaker actually used. The model therefore learned real spoken stress, not dictionary stress.

## Limitations

- Ukrainian only. Latin words are transliterated by the front-end.
- Each take is sampled. Sometimes a take stretches a word or pauses oddly. Generate it
  again with another seed.
- Cloning quality follows the reference: use a clean, single-speaker recording and its exact
  transcript.

## Responsible use

Clone only voices you have permission to use. Audio from `synthesize()` carries the
imperceptible [Perth](https://github.com/resemble-ai/Perth) watermark by default.

## License

Apache-2.0, like NeuTTS-Air and NeuCodec.
