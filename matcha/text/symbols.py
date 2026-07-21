""" from https://github.com/keithito/tacotron

Defines the set of symbols used in text input to the model.
"""
_pad = "_"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = (
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
)


# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

# Special symbol ids
SPACE_ID = symbols.index(" ")

# Language tokens: one per Ghana-speech language id, appended AFTER the base symbols so the
# base phoneme ids stay stable (lets us partial-copy a pretrained embedding table). These are
# injected directly by id at the front of the phoneme sequence (see TextMelDataset), NOT
# matched by the char-level tokenizer, so their string form only needs to be unique.
N_BASE_SYMBOLS = len(symbols)
N_LANGS = 42
symbols = symbols + [f"@lang{i}" for i in range(N_LANGS)]


def lang_token_id(lang_id):
    """Phoneme-sequence id for a given language id (0..N_LANGS-1)."""
    return N_BASE_SYMBOLS + int(lang_id)


# Grapheme (orthographic) characters used across the 42 Ghana-speech languages that are NOT
# already in the IPA-oriented base set — needed when training on raw orthography instead of
# lfn phonemes. Appended AFTER the language tokens so base + lang-token ids stay stable
# (a pretrained IPA/lfn checkpoint partial-copies cleanly; only these rows init fresh).
# Codepoint-ordered for deterministic ids. Includes tone/nasal combining marks (kept as
# tokens), accented vowels, ɩ, Hausa hooks, digits, and a few punctuation marks.
GRAPHEME_EXTRAS = ["(", ")", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "[", "]", "à", "á", "â", "ã", "è", "é", "ê", "ë", "ì", "í", "î", "ï", "ñ", "ò", "ó", "ô", "õ", "ù", "ú", "ÿ", "ā", "ă", "đ", "ē", "ĩ", "ī", "ĺ", "ń", "ō", "ũ", "ū", "ƒ", "ƙ", "ƴ", "ǝ", "ǹ", "ɩ", "̀", "́", "̂", "̃", "̄", "̱", "ḿ", "ẽ", "–", "‘", "’", "ꞌ"]
symbols = symbols + GRAPHEME_EXTRAS
