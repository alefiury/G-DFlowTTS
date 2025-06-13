import re
import os
import logging

import swifter
import torchaudio
import pandas as pd
from tqdm import tqdm

import phonemizer
from unidecode import unidecode

# To avoid excessive logging we set the log level of the phonemizer package to Critical
critical_logger = logging.getLogger("phonemizer")
critical_logger.setLevel(logging.CRITICAL)

# Intializing the phonemizer globally significantly reduces the speed
# now the phonemizer is not initialising at every call
# Might be less flexible, but it is much-much faster
global_phonemizer = phonemizer.backend.EspeakBackend(
    language="en-us",
    preserve_punctuation=True,
    with_stress=True,
    language_switch="remove-flags",
    logger=critical_logger,
)

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# Remove brackets
_brackets_re = re.compile(r"[\[\]\(\)\{\}]")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [
    (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def lowercase(text):
    return text.lower()


def remove_brackets(text):
    return re.sub(_brackets_re, "", text)


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    """Pipeline for non-English text that transliterates to ASCII."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners2(text):
    """Pipeline for English text, including abbreviation expansion. + punctuation + stress"""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    phonemes = global_phonemizer.phonemize([text], strip=True, njobs=1)[0]
    # Added in some cases espeak is not removing brackets
    phonemes = remove_brackets(phonemes)
    phonemes = collapse_whitespace(phonemes)
    return phonemes


def get_audio_info(path: str) -> float:
    """
    Get basic information related to number of frames,
    sample rate and number of channels.
    """
    try:
        info = torchaudio.info(path)
    except Exception as e:
        print(f"Error in {path}: {e}")
        return 0
    return info.num_frames / info.sample_rate


def ljspeech(root_path, meta_file):  # pylint: disable=unused-argument
    """Normalizes the LJSpeech meta data file to TTS format
    https://keithito.com/LJ-Speech-Dataset/"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "ljspeech"
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            audio_token_file = os.path.join(root_path, "tokens", cols[0] + ".pt")
            text = cols[2]
            items.append({"text": text.strip(), "audio_token_file": audio_token_file, "speaker_name": speaker_name, "root_path": root_path})
    return items


def main():
    train_metadata_path = "/hadatasets/alef.ferreira/DATASETS/LibriTTS_R/libri_tts-train-clean-960.csv"
    val_metadata_path = "/hadatasets/alef.ferreira/DATASETS/LibriTTS_R/libri_tts-dev-clean.csv"
    test_metadata_path = "/hadatasets/alef.ferreira/DATASETS/LibriTTS_R/libri_tts-test-clean.csv"

    df_train = pd.read_csv(train_metadata_path)
    df_val = pd.read_csv(val_metadata_path)
    df_test = pd.read_csv(test_metadata_path)

    df_train["language"] = "en-us"
    df_val["language"] = "en-us"
    df_test["language"] = "en-us"

    # df_train["wav_filepath"] = df_train["filepath"].swifter.apply(lambda x: x.replace("LibriTTS_R-wavtokenizer", "LibriTTS_R").replace(".pt", ".wav"))
    # df_val["wav_filepath"] = df_val["filepath"].swifter.apply(lambda x: x.replace("LibriTTS_R-wavtokenizer-dev-clean", "LibriTTS_R-dev-clean").replace(".pt", ".wav"))
    # df_test["wav_filepath"] = df_test["filepath"].swifter.apply(lambda x: x.replace("LibriTTS_R-wavtokenizer-test-clean", "LibriTTS_R-test-clean").replace(".pt", ".wav"))

    # for idx, row in tqdm(df_train.iterrows(), total=len(df_train)):
    #     assert os.path.exists(row["wav_filepath"]), f"File not found: {row['wav_filepath']}"

    # for idx, row in tqdm(df_val.iterrows(), total=len(df_val)):
    #     assert os.path.exists(row["wav_filepath"]), f"File not found: {row['wav_filepath']}"

    # for idx, row in tqdm(df_test.iterrows(), total=len(df_test)):
    #     assert os.path.exists(row["wav_filepath"]), f"File not found: {row['wav_filepath']}"

    df_train["phonemes"] = df_train["text"].swifter.apply(english_cleaners2)
    # df_train["duration"] = df_train["filepath"].swifter.apply(lambda x: get_audio_info(x))

    df_val["phonemes"] = df_val["text"].swifter.apply(english_cleaners2)
    # df_val["duration"] = df_val["filepath"].swifter.apply(lambda x: get_audio_info(x))

    df_test["phonemes"] = df_test["text"].swifter.apply(english_cleaners2)
    # df_test["duration"] = df_test["filepath"].swifter.apply(lambda x: get_audio_info(x))

    df_train = df_train[["filename", "text", "phonemes", "language"]]
    df_val = df_val[["filename", "text", "phonemes", "language"]]
    df_test = df_test[["filename", "text", "phonemes", "language"]]

    df_train.to_csv("/hadatasets/alef.ferreira/DATASETS/LibriTTS_R/libri_tts-train-clean-960-phonemes.csv", index=False)
    df_val.to_csv("/hadatasets/alef.ferreira/DATASETS/LibriTTS_R/libri_tts-dev-clean-phonemes.csv", index=False)
    df_test.to_csv("/hadatasets/alef.ferreira/DATASETS/LibriTTS_R/libri_tts-test-clean-phonemes.csv", index=False)


if __name__ == "__main__":
    main()