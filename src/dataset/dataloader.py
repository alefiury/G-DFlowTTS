import io
import os
import sys
import random

sys.path.append(os.getcwd())

from typing import List, Tuple, Optional

import torch
import torchaudio
import numpy as np
import pandas as pd
import soundfile as sf
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoTokenizer
from utils.phonemes_tokenizer import PhonemeTokenizer


class DurationBPEOfflineDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        base_dir: str,
        filepath_column: str,
        text_tokenizer: AutoTokenizer,
    ):
        """
        data: A list of data entries, each containing 'audio', 'transcription', 'speaker', etc.
        tokenizer: A tokenizer used to convert text into tokens.
        max_audio_duration: Maximum audio duration in seconds (default: 41 seconds).
        """
        self.data = data
        self.base_dir = base_dir
        self.filepath_column = filepath_column
        self.text_tokenizer = text_tokenizer

    def __len__(self):
        return len(self.data)

    def _load_codes(self, filename):
        # treat the case that filename starts with "/"
        if filename.startswith("/") and self.base_dir != "":
            filename = filename[1:]
        codes_path = os.path.join(self.base_dir, filename)

        # if codes_path.endswith(".wav"):
        #     codes_path = codes_path[:-4] + ".pt"

        codes = torch.load(codes_path)

        # remove all empty dimensions
        codes = codes.squeeze()
        return codes

    def __getitem__(self, index):
        datum = self.data.iloc[index]
        transcription = datum["text"]

        if not isinstance(transcription, str):
            transcription = str(transcription)

        filename = datum[self.filepath_column]

        # try:
        audio_codes = self._load_codes(filename)
        tokenized_transcription = torch.tensor(
            self.text_tokenizer.encode(transcription)
        )
        duration = audio_codes.shape[-1]
        # except Exception as e:
        #     print(f"Error loading {filename}: {e}")
        #     next_idx = random.randint(index+1, len(self.data)-1)
        #     return self.__getitem__(next_idx)
        return audio_codes, duration, tokenized_transcription


class DurationBPEOfflineCollateFunc:
    def __init__(
        self,
        audio_pad_token: int = 0,
        audio_bos_token: int = 1,
        text_pad_token: int = 0,
        max_audio_length: Optional[int] = 2048,
    ):
        self.audio_pad_token = audio_pad_token
        self.audio_bos_token = audio_bos_token
        self.text_pad_token = text_pad_token
        self.max_audio_length = max_audio_length

    def __call__(self, batch: List[str]):
        audio_codes, durations, tokenized_transcription_list = zip(*batch)
        B = len(audio_codes)

        processed_audio  = []   # new code sequences with BOS
        dur_list         = []   # durations including BOS
        for codes, dur in zip(audio_codes, durations):
            bos_vec = codes.new_full((1,), self.audio_bos_token)
            codes_bos = torch.cat((bos_vec, codes), dim=0)   # [C, dur+1]
            processed_audio.append(codes_bos)
            dur_list.append(dur + 1)

        Tmax = min(
            max(a.shape[-1] for a in processed_audio),
            self.max_audio_length - 1  # -1 for BOS token
        )

        audio_padded = torch.full(
            (B, Tmax),
            fill_value=self.audio_pad_token,
            dtype=processed_audio[0].dtype
        )
        audio_mask = torch.zeros((B, Tmax), dtype=torch.bool)

        for i, (codes, T) in enumerate(zip(processed_audio, dur_list)):
            T = min(T, Tmax)
            audio_padded[i, :T] = codes[:T]
            audio_mask[i, :T] = 1

        # Text padding
        text_padded = pad_sequence(
            tokenized_transcription_list,
            batch_first=True,
            padding_value=self.text_pad_token
        )
        text_mask = (text_padded != self.text_pad_token)

        remaining_len_pad = torch.zeros((B, Tmax), dtype=torch.long)
        for i, T in enumerate(dur_list):
            T = min(T, Tmax)
            remaining_len_pad[i, :T] = torch.arange(
                T,
                0,
                step=-1,
                dtype=torch.long
            )

        return (
            audio_padded, # [B, Tmax]
            audio_mask, # [B, Tmax]
            torch.tensor(dur_list, dtype=torch.long), # [B]
            text_padded, # [B, Lmax]
            text_mask, # [B, Lmax]
            remaining_len_pad # [B, Tmax]
        )


class HFTextTokenizerDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        base_dir: str,
        filepath_column: str,
    ):
        """
        data: A list of data entries, each containing 'audio', 'transcription', 'speaker', etc.
        tokenizer: A tokenizer used to convert text into tokens.
        max_audio_duration: Maximum audio duration in seconds (default: 41 seconds).
        """
        self.data = data
        self.base_dir = base_dir
        self.filepath_column = filepath_column

    def __len__(self):
        return len(self.data)

    def _load_codes(self, filename):
        # treat the case that filename starts with "/"
        if filename.startswith("/") and self.base_dir != "":
            filename = filename[1:]
        codes_path = os.path.join(self.base_dir, filename)

        if codes_path.endswith(".wav"):
            codes_path = codes_path[:-4] + ".pt"

        codes = torch.load(codes_path)

        # remove all empty dimensions
        codes = codes.squeeze()
        return codes

    def __getitem__(self, index):
        datum = self.data.iloc[index]
        transcription = datum["text"]
        filename = datum[self.filepath_column]
        try:
            audio_codes = self._load_codes(filename)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            next_idx = random.randint(index+1, len(self.data)-1)
            return self.__getitem__(next_idx)

        return audio_codes, transcription


class HFTextTokenizerCollator:
    def __init__(
        self,
        text_tokenizer: AutoTokenizer,
        max_audio_length: int,
        audio_pad_token: int,
        audio_eos_token: int,
        audio_pad_type: str,
        use_eos_as_pad: bool = False,
    ):
        self.text_tokenizer = text_tokenizer

        self.max_audio_length = max_audio_length
        self.audio_pad_token = audio_pad_token
        self.audio_eos_token = audio_eos_token
        self.audio_pad_type = audio_pad_type
        self.use_eos_as_pad = use_eos_as_pad

        if self.use_eos_as_pad:
            print("\n\tUsing EOS token as padding!!!\n")

    def pad_audio_codec(self, feature: torch.Tensor, max_length: int, padding_value: int = 0) -> torch.Tensor:
        """
        Pads the feature tensor along its sequence dimension to max_length.
        Assumes feature has shape (B, L) or (B, L, D).
        """
        current_length = feature.size(1)
        if current_length < max_length:
            pad_amount = max_length - current_length
            padded_feature = F.pad(
                feature,
                (0, pad_amount),
                mode="constant",
                value=padding_value
            )
            return padded_feature
        elif current_length > max_length:
            return feature[:, :max_length]
        else:
            return feature

    def __call__(self, batch: List[str]):
        audio_codes, transcriptions = zip(*batch)

        # Decide target batch length (still +1 for EOS)
        if self.audio_pad_type == "variable":
            max_audio_length = max([audio.shape[-1] + 1 for audio in audio_codes])  # +1 for EOS
            max_audio_length = min(max_audio_length, self.max_audio_length)
        elif self.audio_pad_type == "fixed":
            max_audio_length = self.max_audio_length
        else:
            raise ValueError(f"Unknown audio_pad_type: {self.audio_pad_type}")

        # NEW: effective pad id (EOS or PAD)
        effective_pad_id = self.audio_eos_token if self.use_eos_as_pad else self.audio_pad_token

        padded_audio_list = []
        lengths_with_eos = []  # for attention mask

        for audio in audio_codes:
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)  # [1, L]

            # make room for EOS if we must truncate
            if audio.size(1) >= max_audio_length:
                audio = audio[:, :max_audio_length - 1]

            # append gold EOS
            eos_col = torch.full((audio.size(0), 1), self.audio_eos_token, dtype=audio.dtype, device=audio.device)
            audio = torch.cat([audio, eos_col], dim=1)  # [1, L’]
            lengths_with_eos.append(audio.size(1))      # scalar len including EOS

            # pad/truncate to batch max with chosen effective pad id
            padded_audio = self.pad_audio_codec(audio, max_audio_length, effective_pad_id)
            if padded_audio.size(0) == 1:
                padded_audio = padded_audio.squeeze(0)  # [L]
            padded_audio_list.append(padded_audio)

        audio_codes_padded = torch.stack(padded_audio_list, dim=0)  # [B, L]

        # Build audio attention mask: True up to *gold* (content + EOS), False after
        B, L = audio_codes_padded.shape
        lengths_with_eos = torch.tensor(lengths_with_eos, device=audio_codes_padded.device, dtype=torch.long)
        arangeL = torch.arange(L, device=audio_codes_padded.device).unsqueeze(0)  # [1, L]
        # In this attention mask 1 means valid token (not padding)
        audio_att_mask = (arangeL < lengths_with_eos.unsqueeze(1))  # [B, L] bool

        transcription_encodings = self.text_tokenizer(
            list(transcriptions),
            padding=True,
            return_tensors="pt",
        )

        transcription_padded = transcription_encodings["input_ids"]
        transcription_att_mask = transcription_encodings["attention_mask"].bool()

        x_1 = audio_codes_padded
        x_1_att_mask = audio_att_mask

        return x_1, x_1_att_mask, transcription_padded, transcription_att_mask


class StreamingHFTextTokenizerCollator:
    """Collate streamed Parquet rows into raw 16-kHz utterances + text tokens.

    Codec tokenization intentionally stays out of DataLoader workers and is run
    by the LightningModule on the training device. This avoids loading a large
    NeuCodec model once per worker and preserves per-utterance codec padding.

    Audio is decoded here with soundfile from the raw bytes stored in the
    Parquet ``audio`` column (``Audio(decode=False)``), bypassing torchcodec.
    Rows whose audio cannot be decoded are logged and dropped instead of
    crashing the DataLoader worker. If every row in a batch is dropped the
    collator returns ``None`` and the LightningModule skips that step.
    """

    def __init__(
        self,
        text_tokenizer: AutoTokenizer,
        text_column: str,
        audio_column: str = "audio",
        sampling_rate: int = 16_000,
        max_audio_duration: Optional[float] = None,
        id_column: Optional[str] = "filepath",
    ):
        self.text_tokenizer = text_tokenizer
        self.text_column = text_column
        self.audio_column = audio_column
        self.sampling_rate = sampling_rate
        self.id_column = id_column
        self.max_audio_frames = (
            int(max_audio_duration * sampling_rate)
            if max_audio_duration is not None
            else None
        )

    @staticmethod
    def _decode_audio(audio) -> Tuple[np.ndarray, int]:
        """Return ``(samples[T] or [T, C] float32, sampling_rate)``.

        Accepts the ``Audio(decode=False)`` dict (``bytes``/``path``) and, for
        robustness, an already decoded ``array``/``sampling_rate`` dict.
        """
        if isinstance(audio, dict) and audio.get("array") is not None:
            return (
                np.asarray(audio["array"], dtype=np.float32),
                int(audio["sampling_rate"]),
            )

        if isinstance(audio, dict):
            source = audio.get("bytes")
            if source is None:
                source = audio.get("path")
            if source is None:
                raise ValueError("Audio sample has neither 'bytes' nor 'path'")
        elif isinstance(audio, (bytes, bytearray, str)):
            source = audio
        else:
            raise TypeError(f"Unsupported audio sample type {type(audio)!r}")

        if isinstance(source, (bytes, bytearray)):
            source = io.BytesIO(source)

        data, sr = sf.read(source, dtype="float32", always_2d=False)
        return data, int(sr)

    def _sample_id(self, sample) -> str:
        if self.id_column and sample.get(self.id_column) is not None:
            return str(sample[self.id_column])
        audio = sample.get(self.audio_column)
        if isinstance(audio, dict) and audio.get("path"):
            return str(audio["path"])
        return "<unknown>"

    def _prepare_waveform(self, audio) -> torch.Tensor:
        data, source_sr = self._decode_audio(audio)
        waveform = torch.from_numpy(np.ascontiguousarray(data))

        # soundfile returns [T, C] for multi-channel audio; downmix to mono.
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=1)
        if waveform.ndim != 1:
            raise ValueError(
                f"Expected mono waveform, got shape {tuple(waveform.shape)}"
            )

        if source_sr != self.sampling_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                source_sr,
                self.sampling_rate,
            )

        if self.max_audio_frames is not None:
            waveform = waveform[: self.max_audio_frames]

        if waveform.numel() == 0:
            raise ValueError("Encountered an empty waveform in streamed dataset")

        # [1, T]; Lightning will move tensors inside this list to the device.
        return waveform.unsqueeze(0)

    def __call__(self, batch):
        waveforms = []
        transcriptions = []

        for sample in batch:
            transcription = sample.get(self.text_column)

            # print("="*100)
            # print(transcription)

            if transcription is None or not str(transcription).strip():
                # print("="*100)
                # print(transcription)
                # raise ValueError(
                #     f"Empty transcription in column '{self.text_column}'. "
                #     "Choose a populated text column in the YAML config."
                # )

                transcription = "..."

                print(f"Empty transcription in column '{self.text_column}'.")

            try:
                waveform = self._prepare_waveform(sample[self.audio_column])
            except Exception as exc:  # noqa: BLE001 - any decode failure
                print(
                    "Dropping undecodable audio sample "
                    f"'{self._sample_id(sample)}': {type(exc).__name__}: {exc}"
                )
                continue

            waveforms.append(waveform)
            transcriptions.append(str(transcription))

        if not waveforms:
            print("All samples in this batch were dropped; skipping batch.")
            return None

        transcription_encodings = self.text_tokenizer(
            transcriptions,
            padding=True,
            return_tensors="pt",
        )

        return (
            waveforms,
            transcription_encodings["input_ids"],
            transcription_encodings["attention_mask"].bool(),
        )


class StreamingHFCodesCollator:
    """Collate streamed rows that already hold codec tokens (e.g.
    ``neuphonic/emilia-yodas-english-neucodec``) into training batches.

    Each row is turned into the same ``(codes, text)`` pair returned by
    ``HFTextTokenizerDataset`` and passed to ``HFTextTokenizerCollator``, so
    padding, EOS and attention masks match the offline training path exactly.
    Rows without codes are dropped; if a whole batch is dropped the collator
    returns ``None`` and the LightningModule skips that step.
    """

    def __init__(
        self,
        codes_collator: HFTextTokenizerCollator,
        text_column: str = "text",
        codes_column: str = "codes",
        id_column: Optional[str] = "_id",
    ):
        self.codes_collator = codes_collator
        self.text_column = text_column
        self.codes_column = codes_column
        self.id_column = id_column

    def __call__(self, batch):
        samples = []
        for sample in batch:
            codes = sample.get(self.codes_column)
            if codes is None or len(codes) == 0:
                sample_id = sample.get(self.id_column) if self.id_column else None
                print(f"Dropping sample without codes: '{sample_id}'")
                continue
            samples.append(
                (
                    torch.as_tensor(codes, dtype=torch.long),
                    str(sample.get(self.text_column) or ""),
                )
            )

        if not samples:
            print("All samples in this batch were dropped; skipping batch.")
            return None

        return self.codes_collator(samples)


class PhonemesDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        base_dir: str,
        filepath_column: str,
        text_column: str = "phonemes",
    ):
        """
        data: A list of data entries, each containing 'audio', 'transcription', 'speaker', etc.
        tokenizer: A tokenizer used to convert text into tokens.
        max_audio_duration: Maximum audio duration in seconds (default: 41 seconds).
        """
        self.data = data
        self.base_dir = base_dir
        self.filepath_column = filepath_column
        self.text_column = text_column

    def __len__(self):
        return len(self.data)

    def _load_codes(self, filename):
        # treat the case that filename starts with "/"
        if filename.startswith("/") and self.base_dir != "":
            filename = filename[1:]
        codes_path = os.path.join(self.base_dir, filename)

        if codes_path.endswith(".wav"):
            codes_path = codes_path[:-4] + ".pt"

        codes = torch.load(codes_path)

        # remove all empty dimensions
        codes = codes.squeeze()
        return codes

    def __getitem__(self, index):
        datum = self.data.iloc[index]
        transcription = datum[self.text_column]
        filename = datum[self.filepath_column]
        try:
            audio_codes = self._load_codes(filename)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            next_idx = random.randint(index+1, len(self.data)-1)
            return self.__getitem__(next_idx)

        return audio_codes, transcription


class PhonemeTokenizerCollator:
    def __init__(
        self,
        phoneme_tokenizer: PhonemeTokenizer,
        max_audio_length: int,
        audio_pad_token: int,
        audio_eos_token: int,
        audio_pad_type: str,
        use_eos_as_pad: bool = False,
    ):
        self.phoneme_tokenizer = phoneme_tokenizer

        self.max_audio_length = max_audio_length
        self.audio_pad_token = audio_pad_token
        self.audio_eos_token = audio_eos_token
        self.audio_pad_type = audio_pad_type
        self.use_eos_as_pad = use_eos_as_pad

        if self.use_eos_as_pad:
            print("\n\tUsing EOS token as padding!!!\n")

    def pad_audio_codec(self, feature: torch.Tensor, max_length: int, padding_value: int = 0) -> torch.Tensor:
        """
        Pads the feature tensor along its sequence dimension to max_length.
        Assumes feature has shape (B, L) or (B, L, D).
        """
        current_length = feature.size(1)
        if current_length < max_length:
            pad_amount = max_length - current_length
            padded_feature = F.pad(
                feature,
                (0, pad_amount),
                mode="constant",
                value=padding_value
            )
            return padded_feature
        elif current_length > max_length:
            return feature[:, :max_length]
        else:
            return feature

    def __call__(self, batch: List[str]):
        audio_codes, transcriptions = zip(*batch)

        # Decide target batch length (still +1 for EOS)
        if self.audio_pad_type == "variable":
            max_audio_length = max([audio.shape[-1] + 1 for audio in audio_codes])  # +1 for EOS
            max_audio_length = min(max_audio_length, self.max_audio_length)
        elif self.audio_pad_type == "fixed":
            max_audio_length = self.max_audio_length
        else:
            raise ValueError(f"Unknown audio_pad_type: {self.audio_pad_type}")

        # NEW: effective pad id (EOS or PAD)
        effective_pad_id = self.audio_eos_token if self.use_eos_as_pad else self.audio_pad_token

        padded_audio_list = []
        lengths_with_eos = []  # for attention mask

        for audio in audio_codes:
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)  # [1, L]

            # make room for EOS if we must truncate
            if audio.size(1) >= max_audio_length:
                audio = audio[:, :max_audio_length - 1]

            # append gold EOS
            eos_col = torch.full((audio.size(0), 1), self.audio_eos_token, dtype=audio.dtype, device=audio.device)
            audio = torch.cat([audio, eos_col], dim=1)  # [1, L’]
            lengths_with_eos.append(audio.size(1))      # scalar len including EOS

            # pad/truncate to batch max with chosen effective pad id
            padded_audio = self.pad_audio_codec(audio, max_audio_length, effective_pad_id)
            if padded_audio.size(0) == 1:
                padded_audio = padded_audio.squeeze(0)  # [L]
            padded_audio_list.append(padded_audio)

        audio_codes_padded = torch.stack(padded_audio_list, dim=0)  # [B, L]

        # Build audio attention mask: True up to *gold* (content + EOS), False after
        B, L = audio_codes_padded.shape
        lengths_with_eos = torch.tensor(lengths_with_eos, device=audio_codes_padded.device, dtype=torch.long)
        arangeL = torch.arange(L, device=audio_codes_padded.device).unsqueeze(0)  # [1, L]
        # In this attention mask 1 means valid token (not padding)
        audio_att_mask = (arangeL < lengths_with_eos.unsqueeze(1))  # [B, L] bool

        transcription_encodings = self.phoneme_tokenizer(list(transcriptions))

        transcription_padded = transcription_encodings.input_ids
        transcription_att_mask = transcription_encodings.attention_mask.bool()

        x_1 = audio_codes_padded
        x_1_att_mask = audio_att_mask

        return x_1, x_1_att_mask, transcription_padded, transcription_att_mask
