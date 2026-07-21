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
