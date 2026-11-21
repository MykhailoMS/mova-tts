"""End-to-end with the released model: pytest -m slow (downloads ~1.5 GB on first run)."""
import numpy as np
import pytest


@pytest.mark.slow
def test_stream_and_synthesize_watermarked():
    from mova_tts import MovaTTS
    tts = MovaTTS(seed=3)
    assert tts.watermarker is not None, "Perth watermarking must load from a plain install"
    voice = tts.example_voice()
    chunks = list(tts.stream("Доброго дня! Як ваші справи?", voice))
    assert len(chunks) >= 2 and all(c.dtype == np.float32 for c in chunks)
    wav = np.concatenate(chunks)
    assert 1.0 < len(wav) / 24000 < 6.0 and np.abs(wav).max() > 0.05
    full = tts.synthesize("Доброго дня!", voice)
    assert 0.5 < len(full) / 24000 < 4.0
