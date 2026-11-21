"""Streaming NeuCodec decoding with left context, look-ahead and a crossfade.

Chunk k covers codes [s, e). It is decoded from the window [s - left, e + right) and only the
samples of [s, e) are emitted, plus `xfade` samples of the look-ahead kept to crossfade into
the next chunk. A shorter `first` chunk gives a lower time to first audio. On 120 held-out
sentences this is as good as decoding the whole utterance at once (the stock neutts stream
decoder loses ~0.2 UTMOS).
"""
from __future__ import annotations

import numpy as np

HOP = 480                     # 24 kHz samples per NeuCodec code (50 codes / s)
STREAM_CFG = dict(first=10, chunk=50, left=150, right=25, xfade=960)


class StreamDecoder:
    def __init__(self, decode_fn, chunk=50, first=10, left=150, right=25, xfade=960):
        """decode_fn(np.int64 codes (T,)) -> float32 samples (T * HOP,)"""
        assert xfade <= right * HOP
        self.dec, self.chunk, self.first = decode_fn, chunk, first
        self.left, self.right, self.xfade = left, right, xfade
        self.codes: list[int] = []
        self.s, self.tail = 0, None
        self.fade = np.linspace(0, 1, xfade, dtype=np.float32) if xfade else None

    def _emit(self, e, final=False):
        s = self.s
        a = max(0, s - self.left)
        b = len(self.codes) if final else e + self.right
        w = self.dec(np.asarray(self.codes[a:b], dtype=np.int64))
        o, end = (s - a) * HOP, (e - a) * HOP
        out = w[o:end].copy()
        if self.tail is not None and self.xfade:
            n = min(self.xfade, len(out), len(self.tail))
            out[:n] = self.tail[:n] * (1 - self.fade[:n]) + out[:n] * self.fade[:n]
        self.tail = None if final else w[end:end + self.xfade]
        self.s = e
        return out

    def push(self, codes):
        """Add new codes; returns the audio chunks that are ready (maybe none)."""
        self.codes.extend(int(c) for c in codes)
        out = []
        while True:
            e = self.s + (self.first if self.s == 0 else self.chunk)
            if len(self.codes) < e + self.right:
                break
            out.append(self._emit(e))
        return out

    def flush(self):
        if self.s >= len(self.codes):
            return []
        return [self._emit(len(self.codes), final=True)]
