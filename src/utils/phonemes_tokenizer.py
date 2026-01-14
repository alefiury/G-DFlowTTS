from __future__ import annotations
import re

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Any

import torch


_punctuation = '();:,.!?¡¿—…"«»“” '
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = (
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼᵊʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
)

DEFAULT_PUNCT_SYMBOLS = set(list(_punctuation))
VOCAB_LIST = set(list(_letters) + list(_letters_ipa))

# Remove brackets
_brackets_re = re.compile(r"[\[\]\{\}]")

def remove_brackets(text):
    return re.sub(_brackets_re, "", text)

@dataclass
class BatchEncoding:
    input_ids: torch.LongTensor          # (B, T)
    attention_mask: torch.BoolTensor     # (B, T)
    lengths: torch.LongTensor            # (B,)


class PhonemeTokenizer:
    def __init__(
        self,
        symbols: Iterable[str] = VOCAB_LIST,
        *,
        specials: Tuple[str, str, str] = ("<unk>", "<bos>", "<eos>"),  # (unk, bos, eos) naming
    ):
        self.unk_token, self.bos_token, self.eos_token = specials

        # Use a literal space token as "sp" (already included in DEFAULT_PUNCT_SYMBOLS)
        self.sp_token = " "

        base = set(symbols)
        base.update(DEFAULT_PUNCT_SYMBOLS)

        # Ensure specials are not duplicated inside base
        base.discard(self.unk_token)
        base.discard(self.bos_token)
        base.discard(self.eos_token)

        vocab_list: List[str] = [self.eos_token, self.bos_token, self.unk_token]
        vocab_list += sorted(base)

        self.symbol_to_id = {s: i for i, s in enumerate(vocab_list)}
        self.id_to_symbol = vocab_list

    @property
    def unk_id(self) -> int:
        return self.symbol_to_id[self.unk_token]  # 2

    @property
    def pad_id(self) -> int:
        return self.eos_id  # Using eos as pad token (0)

    @property
    def bos_id(self) -> int:
        return self.symbol_to_id[self.bos_token]  # 1

    @property
    def eos_id(self) -> int:
        return self.symbol_to_id[self.eos_token]  # 0

    @property
    def sp_id(self) -> int:
        return self.symbol_to_id[self.sp_token]   # id of " "

    def tokenize_phoneme_string(self, phonemes: str) -> List[str]:
        return list(phonemes)

    def encode_symbols(self, symbols: Sequence[str]) -> List[int]:
        return [self.symbol_to_id.get(s, self.unk_id) for s in symbols]

    def encode_phonemes(
        self,
        phonemes: str,
        *,
        max_length: Optional[int] = None,
    ) -> List[int]:
        syms = self.tokenize_phoneme_string(phonemes)
        ids = self.encode_symbols(syms)

        ids = ids + [self.eos_id]

        if max_length is not None:
            ids = ids[:max_length]

        return ids

    def pad_batch(
        self,
        sequences: Sequence[Sequence[int]],
        *,
        pad_to_multiple_of: Optional[int] = None,
    ) -> BatchEncoding:
        lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
        max_len = int(lengths.max().item()) if len(sequences) else 0

        if pad_to_multiple_of is not None and max_len > 0:
            m = pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m

        batch = torch.full((len(sequences), max_len), fill_value=self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(sequences), max_len), dtype=torch.bool)

        for i, seq in enumerate(sequences):
            L = len(seq)
            if L == 0:
                continue
            batch[i, :L] = torch.tensor(seq, dtype=torch.long)
            mask[i, :L] = True

        return BatchEncoding(input_ids=batch, attention_mask=mask, lengths=lengths)

    def __call__(
        self,
        texts: Sequence[str],
        *,
        g2p: Optional[Any] = None,  # instance of misaki.en.G2P
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> BatchEncoding:
        seqs: List[List[int]] = []

        if isinstance(texts, str):
            texts = [texts]

        for t in texts:
            if g2p is None:
                phonemes = t
            else:
                phonemes, _ = g2p(t)
            # remove brackets from phonemes
            phonemes = remove_brackets(phonemes)
            seqs.append(
                self.encode_phonemes(
                    phonemes,
                    max_length=max_length,
                )
            )
        return self.pad_batch(seqs, pad_to_multiple_of=pad_to_multiple_of)


def build_default_en_us_vocab() -> PhonemeTokenizer:
    return PhonemeTokenizer(VOCAB_LIST)


if __name__ == "__main__":
    tokenizer = build_default_en_us_vocab()
    sample_text = "hɛˈloʊ, wɜːrld!"
    encoded = tokenizer.encode_phonemes(sample_text)
    print("Encoded:", encoded)
    decoded = [tokenizer.id_to_symbol[i] for i in encoded]
    print("Decoded:", decoded)

    # encode batch example
    texts = ["(hɛˈloʊ)", "[wɜːrld!]", "{ðɪs ɪz ə tɛst.}", "ˈpaθ tə fɒnˌiːmz"]
    batch_encoding = tokenizer(
        texts,
    )
    print("Batch input IDs:", batch_encoding.input_ids)
    print("Batch attention mask:", batch_encoding.attention_mask)
    print("Batch lengths:", batch_encoding.lengths)

    # check for unk tokens (compare with unk_id)
    for i, text in enumerate(texts):
        input_ids = batch_encoding.input_ids[i].tolist()
        print(f"Text: {text}")
        print(f"Input IDs: {input_ids}")
        unk_positions = [idx for idx, id_ in enumerate(input_ids) if id_ == tokenizer.unk_id]
        if unk_positions:
            print(f"  Contains unk tokens at positions: {unk_positions}")
        else:
            print("  No unk tokens found.")

