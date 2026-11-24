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
[neuphonic/neutts-air](https://huggingface.co/neuphonic/neutts-air)** on ~1460 hours of
Ukrainian speech. It generates [NeuCodec](https://huggingface.co/neuphonic/neucodec) tokens
from IPA phonemes, and it clones a voice from a 3–12 second reference recording.

```bash
pip install mova-tts
hf auth login    # NeuCodec is gated: accept the terms on neuphonic/neucodec and neucodec-onnx-decoder once
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

## Preset voices

`mova-tts` bundles four ready-made voices. They are taken from the OpenTTS datasets
(Apache-2.0), and samples are in `samples/`:

| preset | voice | UTMOS | WER | sample |
|---|---|---:|---:|---|
| `tetiana` (default) | female | 3.64 | 0.020 | [samples/tetiana.wav](samples/tetiana.wav) |
| `mykyta` | male | 3.67 | 0.000 | [samples/mykyta.wav](samples/mykyta.wav) |
| `oleksa` | male | 3.41 | 0.020 | [samples/oleksa.wav](samples/oleksa.wav) |
| `lada` | female | 2.47 | 0.029 | [samples/lada.wav](samples/lada.wav) |

```python
from mova_tts import MovaTTS, preset
tts = MovaTTS()
tts.save(tts.synthesize("Доброго дня!", preset("mykyta")), "out.wav")
```

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
| whole utterance, Q8_0 | 0.106 | 0.031 | 3.50 | 0.48 |
| streaming, Q8_0 | 0.115 | 0.033 | 3.41 | 0.48 |
| streaming, Q4_K_M | 0.118 | 0.035 | 3.42 | 0.49 |

Measured through the mova-tts package with one sampling seed; a single run varies by about
±0.01 WER. The previous version (0.1) scored 0.123 / 0.136 / 0.131 WER and 3.46 / 3.37 / 3.33
UTMOS on the same sentences and seed.

WER and CER come from Whisper large-v3 transcripts. UTMOS is a predicted naturalness score from 1 to 5.
Speaker similarity is the ECAPA cosine to the speaker's real recordings.

Speed (streaming, front-end included): first audio after 0.11 s on an RTX 4090 (RTF 0.10),
after 0.9 s on an i9-13900KF CPU with 8 threads (RTF 0.69).

Sample with Hugging Face `generate()` settings (repetition penalty 1.1, top-k 50, top-p 0.8,
temperature 1.0). The mova-tts package does this on top of llama.cpp logits; llama-cpp-python's
built-in sampler gives clearly worse speech with this model.

## Training

- **Base:** [neuphonic/neutts-air](https://huggingface.co/neuphonic/neutts-air), a 0.7B
  Qwen-based speech LM (Apache-2.0). Full fine-tune from the base: 30k steps (LR 4e-5), then
  20k more (LR 1.5e-5) sampling every source without repeats.
- **Data:** ~1460 hours of Ukrainian speech: audiobooks read by professional narrators,
  podcasts, Common Voice, and studio voice recordings. Leading and trailing silence was trimmed,
  utterances with long internal pauses were dropped, every clip scoring below 2.5 UTMOS was
  removed (~280 h), no single narrator has more than 20 hours, and all audio was re-encoded
  with NeuCodec. The training data is not released.
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
