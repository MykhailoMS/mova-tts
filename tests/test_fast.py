import numpy as np

from mova_tts.frontend import clean, destress_clitics, needs_verbalization, to_ipa, translit_latin
from mova_tts.stream import HOP, StreamDecoder
from mova_tts.tts import MovaTTS, Voice


def test_clean_and_latin():
    assert clean("«Так» — сказав  він’") == "\"Так\" - сказав він'"
    assert translit_latin("CRM через Telegram") == "сі-ар-ем через телеграм"
    assert needs_verbalization("о 14:30") and not needs_verbalization("за+мок")


def test_destress_clitics_keeps_content_words_and_user_marks():
    s = "Ві́н на́ сто́лі і́ ті́льки та́м"
    assert destress_clitics(s) == "Ві́н на сто́лі і ті́льки та́м"
    assert destress_clitics(s, user_marked={5}) == "Ві́н на́ сто́лі і ті́льки та́м"


def test_ipa_matches_training_format():
    assert to_ipa("Ізра́їль прово́див гучну́ кампа́нію.") == "izˈrajilʲ prɔˈwɔdɪu̯ ɦʊt͡ʃˈnu kɐmˈpanʲijʊ."


def test_stream_decoder_reassembles_exactly():
    """With a decoder that maps each code to its own samples, the streamed output must equal
    the whole-utterance decode, whatever the chunking."""
    def dec(c):
        return np.repeat(c.astype(np.float32), HOP)
    codes = np.arange(1, 238)
    d = StreamDecoder(dec, chunk=50, first=10, left=150, right=25, xfade=0)
    out = []
    for i in range(0, len(codes), 7):
        out += d.push(codes[i:i + 7])
    out += d.flush()
    assert len(out[0]) == 10 * HOP
    np.testing.assert_array_equal(np.concatenate(out), dec(codes))


def test_pieces_pack_sentences():
    t = MovaTTS.__new__(MovaTTS)
    p = t._pieces("Перше речення. Друге! Третє довше речення тут? Кінець.", max_chars=30)
    assert p == ["Перше речення. Друге!", "Третє довше речення тут?", "Кінець."]


def test_voice_round_trip(tmp_path):
    v = Voice(np.array([1, 2, 65535]), "ipa", "текст")
    v.save(tmp_path / "v.npz")
    w = Voice.load(str(tmp_path / "v.npz"))
    assert w.codes.tolist() == [1, 2, 65535] and w.ipa == "ipa" and w.text == "текст"


def test_prompt_layout_matches_neutts():
    t = MovaTTS.__new__(MovaTTS)
    p = t._prompt("ab", Voice(np.array([5, 7]), "ref"))
    assert p == ("user: Convert the text to speech:<|TEXT_PROMPT_START|>ref ab<|TEXT_PROMPT_END|>\n"
                 "assistant:<|SPEECH_GENERATION_START|><|speech_5|><|speech_7|>")


def test_cuda_init_order(monkeypatch, tmp_path):
    """MovaTTS(device="cuda") builds the decoder from the torch codec inside __init__; that
    must not read attributes __init__ has not set yet (it did: AttributeError '_encoder')."""
    import llama_cpp
    import mova_tts.tts as T

    class FakeLlama:
        def __init__(self, *a, **k):
            pass

    class Codec:
        def decode_code(self, x):
            import torch
            return torch.zeros(1, 1, x.shape[-1] * HOP)

        def to(self, *a):
            return self

    monkeypatch.setattr(llama_cpp, "Llama", FakeLlama)
    monkeypatch.setattr(T, "Frontend", lambda **k: None)
    monkeypatch.setattr(T, "_load_watermarker", lambda: None)
    created = []

    def fake_codec(self):
        if self._encoder is None:
            self._encoder = Codec()
            created.append(1)
        return self._encoder
    monkeypatch.setattr(MovaTTS, "_torch_codec", fake_codec)
    import torch
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **k: self)
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"")
    t = MovaTTS(model=str(gguf), device="cuda")
    assert created == [1] and t._decode(np.array([1, 2, 3])).shape == (3 * HOP,)


class _Backbone:
    """Fake llama.cpp: streams `n` speech tokens, then the end token if `stops`."""

    def __init__(self, n, stops=True):
        self.n, self.stops, self.seeds = n, stops, []

    def tokenize(self, b, add_bos=False, special=True):
        return list(range(len(b) // 4))

    def reset(self):
        pass

    def __call__(self, prompt, max_tokens, stop, stream, seed, **kw):
        self.seeds.append(seed)
        for i in range(min(self.n, max_tokens)):
            yield {"choices": [{"text": f"<|speech_{i}|>", "finish_reason": None}]}
        if not self.stops:
            yield {"choices": [{"text": "", "finish_reason": "length"}]}


def _tts(backbone):
    t = MovaTTS.__new__(MovaTTS)
    t.backbone, t.seed, t.watermarker = backbone, 1, None
    t.frontend = lambda text: "ab"                          # 2 IPA chars -> cap 8 * 2 + 50 = 66
    t._decode = lambda c: np.ones(len(c) * HOP, np.float32)
    return t


def test_runaway_take_is_cut_at_the_cap():
    t = _tts(_Backbone(n=1000, stops=False))
    codes = [c for cs in t._codes("ab", Voice(np.array([1]), "r"), seed=0) for c in cs]
    assert len(codes) == 66 and t.last_ok is False


def test_normal_take_is_not_cut():
    t = _tts(_Backbone(n=40))
    codes = [c for cs in t._codes("ab", Voice(np.array([1]), "r"), seed=0) for c in cs]
    assert len(codes) == 40 and t.last_ok is True


def test_synthesize_retries_a_runaway_with_a_new_seed():
    bb = _Backbone(n=1000, stops=False)
    t = _tts(bb)
    wav = t.synthesize("Текст.", Voice(np.array([1]), "r"), retries=2)
    assert len(bb.seeds) == 3 and len(set(bb.seeds)) == 3 and len(wav) == 66 * HOP


def test_unknown_device_is_rejected():
    import pytest
    with pytest.raises(ValueError, match="device"):
        MovaTTS(device="metal Q8_0")             # was silently treated as CPU
    with pytest.raises(ValueError, match="quant"):
        MovaTTS(quant="Q5")


class _FlakyBackbone(_Backbone):
    """First take stops after 1 code (an early stop), later takes are normal."""

    def __call__(self, prompt, max_tokens, stop, stream, seed, **kw):
        self.seeds.append(seed)
        n = 1 if len(self.seeds) == 1 else self.n
        for i in range(n):
            yield {"choices": [{"text": f"<|speech_{i}|>", "finish_reason": None}]}


def test_early_stop_is_retried_in_synthesize():
    bb = _FlakyBackbone(n=40)
    wav = _tts(bb).synthesize("Текст.", Voice(np.array([1]), "r"))
    assert len(bb.seeds) == 2 and len(wav) == 40 * HOP


def test_early_stop_before_any_audio_is_retried_in_stream():
    bb = _FlakyBackbone(n=40)
    out = list(_tts(bb).stream("Текст.", Voice(np.array([1]), "r")))
    assert len(bb.seeds) == 2 and sum(len(c) for c in out) == 40 * HOP


def test_gated_codec_gives_a_clear_message():
    import pytest
    from mova_tts.tts import _neuphonic

    class GatedRepoError(Exception):
        pass

    def load(repo):
        raise GatedRepoError("401 Client Error")
    with pytest.raises(RuntimeError, match="hf auth login"):
        _neuphonic(load, "neuphonic/neucodec-onnx-decoder")
    with pytest.raises(ValueError):                    # other errors pass through unchanged
        _neuphonic(lambda r: (_ for _ in ()).throw(ValueError("x")), "neuphonic/neucodec")


def test_example_voice_is_packaged():
    import os
    from mova_tts.tts import example_voice_files
    wav, txt = example_voice_files()
    assert os.path.getsize(wav) > 100_000 and "кримського" in open(txt, encoding="utf-8").read()


def test_missing_recording_gives_a_clear_error():
    import pytest
    t = MovaTTS.__new__(MovaTTS)
    with pytest.raises(FileNotFoundError, match="example_voice"):
        t.voice("no-such.wav", "текст")


def test_watermarker_loads():
    from mova_tts.tts import _load_watermarker
    assert _load_watermarker() is not None


def test_wheel_contains_the_example_voice(tmp_path):
    """hatchling honours .gitignore (*.wav): the bundled voice must still be in the wheel."""
    import os
    import subprocess
    import sys
    import zipfile
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                        "-w", str(tmp_path), root], capture_output=True, text=True)
    if r.returncode:
        import pytest
        pytest.skip("cannot build a wheel here: " + r.stderr[-200:])
    whl = next(p for p in os.listdir(tmp_path) if p.endswith(".whl"))
    names = zipfile.ZipFile(tmp_path / whl).namelist()
    assert "mova_tts/data/example_voice.wav" in names and "mova_tts/data/example_voice.txt" in names


import pytest as _pytest


@_pytest.mark.slow
def test_clean_exit_after_synthesis_on_this_machine():
    """llama.cpp must be freed before interpreter teardown: otherwise Metal aborts with exit
    code 134 after a successful run. Runs a real (tiny) process; skipped without the model."""
    import subprocess
    import sys
    import pytest
    code = ("from mova_tts import MovaTTS\n"
            "t = MovaTTS(watermark=False, verbalize=False)\n"
            "t.synthesize('Так.', t.example_voice())\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=900)
    if "GatedRepoError" in r.stderr or "401" in r.stderr:
        pytest.skip("no Hugging Face access to the model here")
    assert r.returncode == 0, r.stderr[-800:]
    assert "NoneType' object is not callable" not in r.stderr


def test_presets_are_bundled_and_valid():
    from mova_tts import preset, presets
    from mova_tts.tts import DEFAULT_PRESET
    assert presets() == ["lada", "mykyta", "oleksa", "tetiana"] and DEFAULT_PRESET in presets()
    for name in presets():
        v = preset(name)
        assert 150 <= len(v.codes) <= 600 and v.codes.max() < 65536 and v.ipa and v.text   # 3-12 s
    assert preset().text == preset(DEFAULT_PRESET).text


def test_unknown_preset_lists_the_available_ones():
    import pytest
    from mova_tts import preset
    with pytest.raises(ValueError, match="tetiana"):
        preset("nobody")
