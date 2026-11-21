"""mova-tts: Ukrainian TTS with voice cloning (NeuTTS-Air fine-tune, GGUF via llama.cpp,
NeuCodec audio codec), streaming or whole-utterance."""
from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from math import gcd
from typing import Iterator

import numpy as np

import warnings as _warnings

# third-party deprecation noise at model load, nothing the user can act on
_warnings.filterwarnings("ignore", message=r".*`local_dir_use_symlinks` argument is deprecated.*")
_warnings.filterwarnings("ignore", message=r".*`resume_download` argument is deprecated.*")
_warnings.filterwarnings("ignore", message=r".*`torch\.jit\.script` is deprecated.*")
from .frontend import Frontend
from .stream import HOP, STREAM_CFG, StreamDecoder

DEFAULT_MODEL = "Mikhailo/mova-tts"
QUANTS = ("Q8_0", "Q4_K_M")
DEVICES = ("cuda", "metal", "cpu")
SR = 24_000                      # output sample rate
SR_IN = 16_000                   # NeuCodec encoder input
N_CTX = 2048
MAX_CODES_PER_CHAR = 8           # runaway guard, see MovaTTS._codes
MIN_CODES_PER_CHAR = 1           # early-stop guard
# Matched to the HF eval (HF generate() merges the checkpoint's generation_config: top_p 0.8,
# repetition_penalty 1.1 over the whole sequence). llama-cpp-python defaults (top_p 0.95,
# min_p 0.05, no penalty) gave WER 0.16-0.20 on held-out sentences, these 0.12.
SAMPLING = dict(temperature=1.0, top_k=50, top_p=0.8, min_p=0.0, repeat_penalty=1.1)
_SPEECH = re.compile(r"<\|speech_(\d+)\|>")
_END = "<|SPEECH_GENERATION_END|>"
_SENT = re.compile(r"(?<=[.!?…])\s+")


@dataclass
class Voice:
    """A reference voice: its NeuCodec codes and the IPA of what is said in it."""
    codes: np.ndarray
    ipa: str
    text: str = ""

    def save(self, path: str):
        np.savez(path, codes=self.codes.astype(np.int64), ipa=self.ipa, text=self.text)

    @classmethod
    def load(cls, path: str) -> "Voice":
        z = np.load(path, allow_pickle=False)
        return cls(z["codes"].astype(np.int64), str(z["ipa"]), str(z["text"]))


def _perf_cores() -> int:
    if sys.platform == "darwin":
        try:
            return int(subprocess.check_output(["sysctl", "-n", "hw.perflevel0.physicalcpu"], text=True))
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
            pass
    return max((os.cpu_count() or 2) // 2, 1)


def _auto_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "metal"
    return "cpu"


def load_audio(path: str, sr: int = SR_IN) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly
    wav, r = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(1)
    if r != sr:
        g = gcd(r, sr)
        wav = resample_poly(wav, sr // g, r // g).astype(np.float32)
    return wav


def trim_silence(wav: np.ndarray, sr: int = SR_IN, thr_db: float = -45.0, pad: float = 0.12):
    hop = sr // 100
    n = len(wav) // hop
    if n < 3:
        return wav
    e = 10 * np.log10(np.mean(wav[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-10)
    on = np.flatnonzero(e > e.max() + thr_db)
    if not len(on):
        return wav
    p = int(pad * sr)
    return wav[max(0, on[0] * hop - p): min(len(wav), (on[-1] + 1) * hop + p)]


def example_voice_files() -> tuple[str, str]:
    """(wav, transcript) paths of the bundled example voice (Common Voice, CC0)."""
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    return os.path.join(d, "example_voice.wav"), os.path.join(d, "example_voice.txt")


_PRESET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "voices")
DEFAULT_PRESET = "tetiana"          # the cleanest of the bundled voices (UTMOS 3.64)


def presets() -> list[str]:
    """Names of the bundled ready-made voices (already encoded: no recording, no codec encoder)."""
    if not os.path.isdir(_PRESET_DIR):
        return []
    return sorted(f[:-4] for f in os.listdir(_PRESET_DIR) if f.endswith(".npz"))


def preset(name: str = DEFAULT_PRESET) -> Voice:
    """A bundled voice by name, see presets(). Voices from the OpenTTS datasets (Apache-2.0):
    tetiana and lada (female), mykyta and oleksa (male)."""
    p = os.path.join(_PRESET_DIR, f"{name}.npz")
    if not os.path.isfile(p):
        raise ValueError(f"unknown preset {name!r}; available: {', '.join(presets())}")
    return Voice.load(p)


def _neuphonic(load, repo):
    """NeuCodec repos on HF are gated (approved at once, but you must accept the terms)."""
    import warnings
    try:
        with warnings.catch_warnings():      # neucodec passes deprecated hf_hub_download args
            warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
            warnings.filterwarnings("ignore", category=FutureWarning)
            return load(repo)
    except Exception as e:
        if type(e).__name__ in ("GatedRepoError", "HTTPStatusError") or "401" in str(e) or "403" in str(e):
            raise RuntimeError(
                f"{repo} is a gated Hugging Face repo. Once: open https://huggingface.co/{repo} and "
                "https://huggingface.co/neuphonic/neucodec, accept the terms (approved at once), "
                "then log in with `hf auth login`.") from e
        raise


class MovaTTS:
    """Ukrainian text-to-speech with voice cloning.

    model:   HF repo id or a local dir with gguf/mova-tts-<quant>.gguf, or a .gguf path
    quant:   "Q8_0" (default, best) or "Q4_K_M" (smaller, ~same quality)
    device:  "auto" | "cuda" | "metal" (Apple GPU) | "cpu"
    watermark: add the Perth imperceptible watermark to synthesize() output (default on)
    seed:    fixed seed for repeatable takes; None = a new random take every call
    stress_model / user_dict: passed to naholos (see the naholos README)
    verbalize: numbers -> words (default on; the model cannot read digits)
    """

    def __init__(self, model: str = DEFAULT_MODEL, quant: str = "Q8_0", device: str = "auto",
                 watermark: bool = True, seed: int | None = None, stress_model: str | None = None,
                 user_dict=None, verbalize: bool = True, threads: int | None = None):
        # torch before llama_cpp: a CUDA llama.cpp build loads its own libcudart first, and a
        # newer torch then fails with "undefined symbol cudaGetDriverEntryPointByVersion"
        import torch  # noqa: F401
        from llama_cpp import Llama
        if device not in ("auto",) + DEVICES:
            raise ValueError(f"device must be 'auto' or one of {DEVICES}, not {device!r}")
        if quant not in QUANTS:
            raise ValueError(f"quant must be one of {QUANTS}, not {quant!r}")
        self.device = _auto_device() if device == "auto" else device
        self.seed = seed
        gpu = self.device in ("cuda", "metal")
        threads = threads or _perf_cores()
        if not gpu:
            # torch's default (every logical core) fights llama.cpp's threads: the stress tagger
            # took 823 ms instead of 261 ms per sentence on an i9-13900KF
            torch.set_num_threads(min(torch.get_num_threads(), threads))
        kw = dict(verbose=False, n_gpu_layers=-1 if gpu else 0, n_ctx=N_CTX, n_batch=512,
                  n_ubatch=512, n_threads=threads, n_threads_batch=threads if not gpu else os.cpu_count(),
                  flash_attn=gpu, offload_kqv=gpu)   # a CUDA build otherwise keeps the KV cache on the GPU
        if model.endswith(".gguf") and os.path.isfile(model):
            self.backbone = Llama(model_path=model, **kw)
        elif os.path.isdir(model):
            self.backbone = Llama(model_path=os.path.join(model, "gguf", f"mova-tts-{quant}.gguf"), **kw)
        else:
            self.backbone = Llama.from_pretrained(repo_id=model, filename=f"gguf/mova-tts-{quant}.gguf", **kw)
        self.backbone.last_n_tokens_size = N_CTX           # repeat-penalty window = whole sequence
        # free llama.cpp while the interpreter is still alive: freed during teardown, it hits
        # half-destroyed ctypes ("'NoneType' object is not callable") and the Metal backend
        # then aborts the process (GGML_ASSERT in ggml_metal_device_free, exit code 134)
        import atexit
        import weakref
        ref = weakref.ref(self)
        atexit.register(lambda: ref() is not None and ref().close())
        torch_dev = {"cuda": "cuda", "metal": "mps"}.get(self.device, "cpu")
        self.frontend = Frontend(device=torch_dev, stress_model=stress_model, verbalize=verbalize,
                                 user_dict=user_dict)
        self._encoder = None
        self._decode = self._load_decoder()
        self.watermarker = _load_watermarker() if watermark else None

    # ------------------------------------------------------------------ codec
    def _load_decoder(self):
        if self.device == "cuda":
            import torch
            codec = self._torch_codec()

            @torch.no_grad()
            def dec(c):
                x = torch.from_numpy(c)[None, None].to("cuda")
                return codec.decode_code(x)[0, 0].float().cpu().numpy()
            return dec
        from neucodec import NeuCodecOnnxDecoder         # CPU / Mac: ONNX decoder is faster
        onnx = _neuphonic(NeuCodecOnnxDecoder.from_pretrained, "neuphonic/neucodec-onnx-decoder")
        return lambda c: onnx.decode_code(c.astype(np.int32)[None, None])[0, 0].astype(np.float32)

    def _torch_codec(self):
        if self._encoder is None:
            from neucodec import NeuCodec
            self._encoder = _neuphonic(NeuCodec.from_pretrained, "neuphonic/neucodec").eval()
            if self.device == "cuda":
                self._encoder = self._encoder.to("cuda")
        return self._encoder

    # ------------------------------------------------------------------ voices
    def voice(self, wav_path: str, text: str, max_seconds: float = 12.0) -> Voice:
        """Make a voice from a clean recording (3-12 s) and the exact text spoken in it.
        Save it with voice.save("me.npz") and reuse it with Voice.load("me.npz")."""
        import torch
        if not os.path.isfile(wav_path):
            raise FileNotFoundError(f"no such recording: {wav_path!r} (or try tts.example_voice())")
        w = trim_silence(load_audio(wav_path))[: int(max_seconds * SR_IN)]
        codec = self._torch_codec()
        with torch.no_grad():
            x = torch.from_numpy(w)[None, None].to(next(codec.parameters()).device)
            codes = codec.encode_code(x).cpu().numpy()[0, 0].astype(np.int64)
        return Voice(codes, self.frontend(text), text)

    def example_voice(self) -> Voice:
        """A male voice from Common Voice (CC0), shipped with the package, for a first try."""
        wav, txt = example_voice_files()
        return self.voice(wav, open(txt, encoding="utf-8").read().strip())

    # ------------------------------------------------------------------ synthesis
    def _prompt(self, ipa: str, voice: Voice) -> str:
        codes = "".join(f"<|speech_{int(i)}|>" for i in voice.codes)
        return (f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{voice.ipa} {ipa}"
                f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes}")

    def _codes(self, ipa: str, voice: Voice, seed: int, **sampling) -> Iterator[list[int]]:
        """Stream generated codes. Afterwards self.last_ok is False if the take failed:
        it ran away (no stop by MAX_CODES_PER_CHAR codes per IPA character; real speech needs
        at most 7.95 on 431 held-out sentences, median 3) and was cut, or it stopped too early
        (fewer than MIN_CODES_PER_CHAR; real speech needs at least 2.1)."""
        prompt = self._prompt(ipa, voice)
        n = len(self.backbone.tokenize(prompt.encode(), add_bos=False, special=True))
        if n >= N_CTX - 50:
            raise ValueError("reference voice + text too long for the context: use a shorter "
                             "reference or shorter text")
        cap = min(N_CTX - n, MAX_CODES_PER_CHAR * len(ipa) + 50)
        self.backbone.reset()
        self.last_ok, got = True, 0
        for item in self.backbone(prompt, max_tokens=N_CTX - n, stop=[_END], stream=True,
                                  seed=seed, **{**SAMPLING, **sampling}):
            ids = _SPEECH.findall(item["choices"][0]["text"])
            if ids:
                ids = ids[:cap - got]
                got += len(ids)
                yield [int(i) for i in ids]
            if got >= cap:
                self.last_ok = False
                return
            if item["choices"][0].get("finish_reason") == "length":
                self.last_ok = False
        if got < MIN_CODES_PER_CHAR * len(ipa):
            self.last_ok = False

    def _pieces(self, text: str, max_chars: int):
        """Sentences, packed into pieces of <= max_chars (each piece is one generation)."""
        out, cur = [], ""
        for s in _SENT.split(text.strip()):
            if cur and len(cur) + 1 + len(s) > max_chars:
                out.append(cur)
                cur = s
            else:
                cur = f"{cur} {s}".strip()
        if cur:
            out.append(cur)
        return out

    def _seed(self):
        return self.seed if self.seed is not None else int.from_bytes(os.urandom(4), "little")

    def stream(self, text: str, voice: Voice, max_chars: int = 220, retries: int = 2,
               **sampling) -> Iterator[np.ndarray]:
        """Yield float32 24 kHz audio chunks as they are generated (the first, 0.2 s long,
        after 35 codes).
        Long text is generated sentence by sentence. A take that fails before its first chunk
        is out (an early stop) is regenerated; once audio is out it cannot be taken back, so a
        later runaway is only cut. Stream chunks are not watermarked."""
        seed = self._seed()
        for k, piece in enumerate(self._pieces(text, max_chars)):
            yield from self._stream_ipa(self.frontend(piece), voice, seed + k, retries, **sampling)

    def _stream_ipa(self, ipa, voice, seed, retries=2, **sampling):
        for r in range(retries + 1):
            dec = StreamDecoder(self._decode, **STREAM_CFG)
            emitted = False
            for codes in self._codes(ipa, voice, seed + 7919 * r, **sampling):
                out = dec.push(codes)
                emitted = emitted or bool(out)
                yield from out
            if self.last_ok or emitted or r == retries:
                yield from dec.flush()
                return

    def _synth_ipa(self, ipa, voice, seed, retries=2, **sampling) -> np.ndarray:
        for r in range(retries + 1):
            codes = [c for cs in self._codes(ipa, voice, seed + 7919 * r, **sampling) for c in cs]
            if self.last_ok:
                break
        return self._decode(np.asarray(codes, dtype=np.int64)) if codes else np.zeros(0, np.float32)

    def synthesize(self, text: str, voice: Voice, max_chars: int = 220, pause: float = 0.15,
                   retries: int = 2, **sampling) -> np.ndarray:
        """Whole text -> float32 24 kHz waveform (watermarked unless watermark=False).
        A failed take (runaway or early stop) is regenerated with another seed, up to `retries`."""
        seed, parts = self._seed(), []
        gap = np.zeros(int(pause * SR), np.float32)
        for k, piece in enumerate(self._pieces(text, max_chars)):
            w = self._synth_ipa(self.frontend(piece), voice, seed + k, retries, **sampling)
            if len(w):
                parts += [w, gap]
        wav = np.concatenate(parts[:-1]) if parts else np.zeros(HOP, np.float32)
        if self.watermarker is not None:
            wav = self.watermarker.apply_watermark(wav, sample_rate=SR).astype(np.float32)
        return wav

    def save(self, wav: np.ndarray, path: str):
        import soundfile as sf
        sf.write(path, wav, SR)

    def close(self):
        """Free the llama.cpp model (also done automatically at exit)."""
        bb = getattr(self, "backbone", None)
        if bb is not None and hasattr(bb, "close"):
            bb.close()
        self.backbone = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _load_watermarker():
    try:
        try:
            import pkg_resources  # noqa: F401
        except ModuleNotFoundError:          # resemble-perth needs pkg_resources (old setuptools)
            import importlib.resources
            import types
            shim = types.ModuleType("pkg_resources")
            shim.resource_filename = lambda p, r: str(importlib.resources.files(p) / r)
            sys.modules["pkg_resources"] = shim
        # import the class directly: perth/__init__ swallows the ImportError and sets it to None
        from perth.perth_net.perth_net_implicit.perth_watermarker import PerthImplicitWatermarker
        return PerthImplicitWatermarker()
    except (ImportError, AttributeError, TypeError) as e:
        import warnings
        warnings.warn(f"Perth watermarking unavailable ({e}); audio is not watermarked.")
        return None


def timed_stream(tts: MovaTTS, text: str, voice: Voice):
    """(time to first chunk s, total s, waveform) - for benchmarks."""
    t0 = time.perf_counter()
    first, chunks = None, []
    for c in tts.stream(text, voice):
        if first is None:
            first = time.perf_counter() - t0
        chunks.append(c)
    return first, time.perf_counter() - t0, np.concatenate(chunks)
