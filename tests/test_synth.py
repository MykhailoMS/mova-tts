"""End-to-end with the released model: pytest -m slow (downloads ~1.5 GB on first run)."""
import os

import numpy as np
import pytest

EX = os.path.join(os.path.dirname(__file__), "..", "examples")


@pytest.mark.slow
def test_stream_and_synthesize():
    from mova_tts import MovaTTS
    tts = MovaTTS(seed=3)
    voice = tts.voice(os.path.join(EX, "voice_cv.wav"), open(os.path.join(EX, "voice_cv.txt")).read())
    chunks = list(tts.stream("Доброго дня! Як ваші справи?", voice))
    assert len(chunks) >= 2 and all(c.dtype == np.float32 for c in chunks)
    wav = np.concatenate(chunks)
    assert 1.0 < len(wav) / 24000 < 6.0 and np.abs(wav).max() > 0.05
    full = tts.synthesize("Доброго дня!", voice)
    assert 0.5 < len(full) / 24000 < 4.0
