"""Text front-end: raw Ukrainian text -> phonemes (IPA) the model was trained on.

  1. clean        NFC, apostrophes, quotes, dashes, spaces
  2. verbalize    numbers, dates, %, currency -> words with skypro1111/m2m100-ukr-verbalization
                  (CTranslate2; downloaded the first time a text needs it)
  3. latin        Latin words -> Cyrillic (acronyms by letter names: CRM -> сі-ар-ем)
  4. stress       naholos (dictionary + ByT5 tagger). Your own marks win: за́мок or за+мок
  5. ipa          ipa-uk (Wiktionary uk-pronunciation algorithm), punctuation kept
"""
from __future__ import annotations

import re
import unicodedata

from . import _ipa_uk

ACUTE = "\u0301"

_NEEDS_VERB = re.compile(r"[0-9%№$€₴§@&=/]|\b(грн|км|кг|см|мм|млн|млрд|тис|ст|р)\.")
_APOS = re.compile(r"[’ʼ`‘]")
_QUOTES = re.compile(r"[«»„“”]")
_DASH = re.compile(r"[—–]")
_WS = re.compile(r"\s+")

_LETTER_NAMES = {  # English letter names as Ukrainians say them in acronyms (CRM, API, IT)
    "a": "ей", "b": "бі", "c": "сі", "d": "ді", "e": "і", "f": "еф", "g": "джі", "h": "ейч",
    "i": "ай", "j": "джей", "k": "кей", "l": "ел", "m": "ем", "n": "ен", "o": "оу", "p": "пі",
    "q": "к'ю", "r": "ар", "s": "ес", "t": "ті", "u": "ю", "v": "ві", "w": "дабл-ю", "x": "екс",
    "y": "вай", "z": "зед"}
_LAT_DI = [("sh", "ш"), ("ch", "ч"), ("th", "т"), ("ph", "ф"), ("oo", "у"), ("ee", "і"),
           ("ck", "к"), ("qu", "кв"), ("ts", "ц"), ("ya", "я"), ("yu", "ю")]
_LAT_1 = {"a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф", "g": "г", "h": "х",
          "i": "і", "j": "дж", "k": "к", "l": "л", "m": "м", "n": "н", "o": "о", "p": "п",
          "q": "к", "r": "р", "s": "с", "t": "т", "u": "у", "v": "в", "w": "в", "x": "кс",
          "y": "і", "z": "з", "'": ""}
_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z']*")
_LATIN_WORDS = {"rest": "рест", "api": "ей-пі-ай", "online": "онлайн", "email": "імейл",
                "google": "гугл", "telegram": "телеграм", "python": "пайтон", "cloud": "клауд"}

# One-vowel function words carry no stress in the training labels (they form one stress
# domain with the next word); every other one-vowel word is stressed (ві́н, та́м).
CLITICS = {
    "в", "у", "з", "із", "зі", "зо", "на", "до", "за", "по", "про", "над", "під", "при", "без",
    "для", "від", "од", "як", "о", "об", "перед", "поза", "повз", "крізь", "через", "між",
    "серед", "біля", "коло", "попри", "щодо", "проти", "навколо", "довкола", "заради",
    "ради", "замість", "окрім", "крім", "поміж", "понад",
    "і", "й", "та", "а", "але", "бо", "що", "щоб", "аби", "чи", "коли", "якщо", "хоч",
    "хоча", "ніж", "ані", "ну", "то", "же", "аж",
    "не", "ні", "би", "б", "ж", "хай", "нехай", "лиш", "лише", "тільки", "навіть",
    "ось", "от", "он", "це", "цей",
    "я", "ти", "ми", "ви", "мене", "тебе", "його", "її", "їх", "їм", "нам", "вам", "мені",
    "тобі", "собі", "се", "сей",
}
_VOWELS = set("аеиіоуяюєїАЕИІОУЯЮЄЇ")
_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*", re.U)


def destress_clitics(stressed: str, user_marked: set[int] = frozenset()) -> str:
    """Drop the mark from one-vowel clitics, unless the user put it there."""
    plain, out = stressed.replace(ACUTE, ""), []
    marks, i = set(), -1
    for ch in stressed:
        if ch == ACUTE:
            marks.add(i)
        else:
            i += 1
    drop = set()
    for m in _WORD.finditer(plain):
        w = m.group(0)
        if w.lower() in CLITICS and sum(c in _VOWELS for c in w) == 1:
            drop |= {j for j in range(m.start(), m.end()) if j in marks and j not in user_marked}
    for j, ch in enumerate(plain):
        out.append(ch + (ACUTE if j in marks and j not in drop else ""))
    return "".join(out)


def clean(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _APOS.sub("'", text)
    text = _QUOTES.sub('"', text)
    text = _DASH.sub("-", text)
    return _WS.sub(" ", text).strip()


def needs_verbalization(text: str) -> bool:
    return bool(_NEEDS_VERB.search(text))


def translit_latin(text: str) -> str:
    """Latin words -> Cyrillic, so the model never sees symbols it was not trained on.
    All-caps 2-6 letters = acronym spelled by letter names; a few known words; otherwise a
    rough letter-by-letter transliteration."""
    def one(m):
        w = m.group(0)
        low = w.lower()
        if low in _LATIN_WORDS:
            return _LATIN_WORDS[low]
        if w.isupper() and 2 <= len(w) <= 6:
            return "-".join(_LETTER_NAMES[c] for c in low if c in _LETTER_NAMES)
        out, i = [], 0
        while i < len(low):
            for a, b in _LAT_DI:
                if low.startswith(a, i):
                    out.append(b)
                    i += len(a)
                    break
            else:
                out.append(_LAT_1.get(low[i], ""))
                i += 1
        return "".join(out)
    return _LATIN_WORD.sub(one, text)


def to_ipa(stressed: str) -> str:
    """Stressed text (U+0301 after the stressed vowel) -> IPA, punctuation kept."""
    return _WS.sub(" ", _ipa_uk.ipa(stressed.replace("\u00b4", ACUTE), False)).strip()


class Verbalizer:
    """Numbers, dates, units -> words (CTranslate2 m2m100)."""

    def __init__(self, device: str = "cpu"):
        import ctranslate2
        from transformers import M2M100Tokenizer
        from huggingface_hub import snapshot_download
        path = snapshot_download("skypro1111/m2m100-ukr-verbalization-ct2")
        self.tok = M2M100Tokenizer.from_pretrained("skypro1111/m2m100-ukr-verbalization")
        self.tok.src_lang = "uk"
        device = "cuda" if device == "cuda" else "cpu"          # CTranslate2: no MPS
        ct = "int8_float16" if device == "cuda" else "int8"
        self.tr = ctranslate2.Translator(path, device=device, compute_type=ct)
        self.prefix = [self.tok.lang_code_to_token["uk"]]

    def __call__(self, text: str) -> str:
        src = self.tok.convert_ids_to_tokens(self.tok.encode(text))
        r = self.tr.translate_batch([src], target_prefix=[self.prefix], beam_size=1,
                                    max_decoding_length=512)[0]
        return self.tok.decode(self.tok.convert_tokens_to_ids(r.hypotheses[0][1:]))


class Frontend:
    """text -> IPA. verbalize=False keeps digits as they are (the model cannot read them)."""

    def __init__(self, device: str = "cpu", stress_model: str | None = None,
                 verbalize: bool = True, user_dict=None):
        self.device, self.verbalize = device, verbalize
        self._verb = None
        from naholos import DEFAULT_MODEL, Naholos
        self.stress = Naholos(model=stress_model or DEFAULT_MODEL, device=device, user_dict=user_dict,
                              monosyllables=True)

    def stressed(self, text: str) -> str:
        """Cleaned, verbalized, stressed text (what is turned into IPA)."""
        from naholos import user_marks
        text = clean(text)
        if self.verbalize and needs_verbalization(text):
            if self._verb is None:
                self._verb = Verbalizer(self.device)
            text = clean(self._verb(text))
        text = translit_latin(text)
        _, mine = user_marks(text, self.stress.plus_marks)
        return destress_clitics(self.stress(text), mine)

    def __call__(self, text: str) -> str:
        return to_ipa(self.stressed(text))
