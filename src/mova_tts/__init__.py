"""mova-tts - Ukrainian TTS with voice cloning (NeuTTS-Air fine-tune)."""
from .frontend import Frontend
from .tts import DEFAULT_MODEL, SR, MovaTTS, Voice, preset, presets

__version__ = "0.1.3"
__all__ = ["MovaTTS", "Voice", "Frontend", "DEFAULT_MODEL", "SR", "preset", "presets"]
