"""mova-tts "Текст" --ref me.wav --ref-text "що сказано в me.wav" -o out.wav"""
import argparse
import sys

from .tts import DEFAULT_MODEL, QUANTS, MovaTTS, Voice, timed_stream


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mova-tts", description="Ukrainian text-to-speech with voice cloning")
    ap.add_argument("text", nargs="*", help="text to speak; read from stdin if omitted")
    ap.add_argument("-o", "--out", default="out.wav")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--ref", help="reference recording of the voice (3-12 s, clean)")
    g.add_argument("--voice", help="a voice saved earlier with --save-voice (.npz); "
                                   "without --ref and --voice the bundled example voice is used")
    ap.add_argument("--ref-text", help="exact text spoken in --ref")
    ap.add_argument("--save-voice", metavar="NPZ", help="save the encoded reference for reuse")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--quant", choices=QUANTS, default="Q8_0")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "metal", "cpu"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--user-dict", help="naholos user dictionary: one stressed word per line")
    ap.add_argument("--no-watermark", action="store_true")
    ap.add_argument("--stream", action="store_true", help="generate in streaming mode, print latency")
    ap.add_argument("--ipa", action="store_true", help="only print the stressed text and IPA")
    a = ap.parse_args(argv)
    if a.ref and not a.ref_text:
        ap.error("--ref needs --ref-text (the exact words spoken in the recording)")
    text = " ".join(a.text) if a.text else sys.stdin.read()
    tts = MovaTTS(model=a.model, quant=a.quant, device=a.device, seed=a.seed, user_dict=a.user_dict,
                  watermark=not a.no_watermark)
    if a.ipa:
        print(tts.frontend.stressed(text))
        print(tts.frontend(text))
        return
    voice = (tts.voice(a.ref, a.ref_text) if a.ref else Voice.load(a.voice) if a.voice
             else tts.example_voice())
    if a.save_voice:
        voice.save(a.save_voice)
    if a.stream:
        first, total, wav = timed_stream(tts, text, voice)
        print(f"first audio {first * 1000:.0f} ms, {len(wav) / 24000:.1f} s of audio in {total:.1f} s",
              file=sys.stderr)
    else:
        wav = tts.synthesize(text, voice)
    tts.save(wav, a.out)
    print(a.out)


if __name__ == "__main__":
    main()
