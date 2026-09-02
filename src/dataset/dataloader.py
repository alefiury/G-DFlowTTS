import os
import sys
import random

sys.path.append(os.getcwd())

from typing import List, Tuple, Optional

import torch
import torchaudio
import pandas as pd
import torch.nn.functional as F
from transformers import AutoFeatureExtractor
from torch.nn.utils.rnn import pad_sequence
from xcodec2.modeling_xcodec2 import XCodec2Model

from transformers import AutoTokenizer
from utils.tokenizer import VoiceBpeTokenizer
from utils.phonemes_tokenizer import PhonemeTokenizer
from utils.symbols import text_to_sequence


class DynamicSingleSpeakerDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        base_dir: str,
        text_tokenizer: VoiceBpeTokenizer,
        sampling_rate: int = 16000,
        max_audio_duration: float = 41.0,
        speech_processor: AutoFeatureExtractor = None
    ):
        """
        data: A list of data entries, each containing 'audio', 'transcription', 'speaker', etc.
        tokenizer: A tokenizer used to convert text into tokens.
        max_audio_duration: Maximum audio duration in seconds (default: 41 seconds).
        """
        self.data = data
        self.base_dir = base_dir
        self.text_tokenizer = text_tokenizer
        self.sampling_rate = sampling_rate
        self.max_audio_frames = int(max_audio_duration * self.sampling_rate)  # Maximum number of frames for the given max duration

        self.speech_processor = speech_processor

    def __len__(self):
        # Each record corresponds to one sample
        return len(self.data)

    def _load_audio(self, filename):
        audio_path = os.path.join(self.base_dir, filename)

        # check if audio_path has .wav extension
        if not audio_path.endswith(".wav"):
            audio_path += ".wav"

        audio, sr = torchaudio.load(audio_path)

        if sr != self.sampling_rate:
            audio = torchaudio.transforms.Resample(sr, self.sampling_rate)(audio)
            sr = self.sampling_rate

        return audio, sr

    def _crop_audio(self, audio):
        audio_length_in_frames = audio.shape[-1]
        if audio_length_in_frames > self.max_audio_frames:
            audio = audio[:, : self.max_audio_frames]
        return audio

    def __getitem__(self, index):
        datum = self.data.iloc[index]
        transcription = datum["transcription"]
        filename = datum["filename"]
        language = datum["language"]

        audio, sr = self._load_audio(filename)
        audio = self._crop_audio(audio)

        tokenized_transcription = self.text_tokenizer.encode(
            transcription,
            lang=language
        )

        audio_pad = F.pad(audio, (160, 160))

        audio_features = self.speech_processor(
            audio_pad,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
        ).data["input_features"]

        return audio, audio_features, torch.tensor(tokenized_transcription)


class DynamicSingleSpeakerCollateFunc:
    def __call__(self, batch: List[str]):
        audio_list, audio_features_list, tokenized_transcription_list = zip(*batch)
        batch_size = len(audio_list)

        # Get max lengths for padding
        max_audio_length = max([audio.shape[-1] for audio in audio_list])
        max_audio_features_length = int(max_audio_length / 320)
        max_audio_length = max_audio_features_length * 320

        audio_padded = torch.zeros(batch_size, 1, max_audio_length)
        for i, audio in enumerate(audio_list):
            padding = max_audio_length - audio.shape[-1]
            if padding > 0:
                audio_padded[i, :, : audio.shape[-1]] = audio
            else:
                audio_padded[i, :, : max_audio_length] = audio[:, :max_audio_length]

        audio_features_padded = torch.zeros(batch_size, 1, max_audio_features_length, 160)
        for i, audio_features in enumerate(audio_features_list):
            padding = max_audio_features_length - audio_features.shape[1]
            if padding > 0:
                audio_features_padded[i, :, : audio_features.shape[1], :] = audio_features
            else:
                audio_features_padded[i, :, : max_audio_features_length, :] = audio_features[:, :max_audio_features_length, :]

        transcription_padded = pad_sequence(tokenized_transcription_list, batch_first=True)

        return audio_padded, audio_features_padded, transcription_padded


class OfflineMultipleSpeakerDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        base_dir: str,
        filepath_column: str,
        text_tokenizer: VoiceBpeTokenizer,
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
        language = datum["language"]
        try:
            audio_codes = self._load_codes(filename)
            tokenized_transcription = torch.tensor(
                self.text_tokenizer.encode(
                    transcription,
                    lang=language
                )
            )
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            next_idx = random.randint(index+1, len(self.data)-1)
            return self.__getitem__(next_idx)

        return audio_codes, tokenized_transcription


class OfflineMultipleSpeakerUniformCollateFunc:
    def __init__(
        self,
        padding_type: str = "batch", # can be "batch" or "max_seq"
        audio_pad_token: int = 0,
        text_pad_token: int = 0,
        max_audio_length: Optional[int] = 2048,
    ):
        self.audio_pad_token = audio_pad_token
        self.text_pad_token = text_pad_token
        self.padding_type = padding_type
        self.max_audio_length = max_audio_length
        assert padding_type in ["batch", "max_seq"], f"Unknown padding type: {padding_type}"

    def __call__(self, batch: List[str]):
        audio_codes, tokenized_transcription_list = zip(*batch)

        if self.padding_type == "batch":
            audio_codes_padded = pad_sequence(
                audio_codes,
                batch_first=True,
                padding_value=self.audio_pad_token
            )
        elif self.padding_type == "max_seq":
            audio_codes_padded = torch.zeros(len(audio_codes), self.max_audio_length)
            for i, audio in enumerate(audio_codes):
                padding = self.max_audio_length - audio.shape[-1]
                if padding > 0:
                    audio_codes_padded[i, : audio.shape[-1]] = audio
                else:
                    audio_codes_padded[i, : self.max_audio_length] = audio[:, :self.max_audio_length]

        transcription_padded = pad_sequence(
            tokenized_transcription_list,
            batch_first=True,
            padding_value=self.text_pad_token
        )

        return audio_codes_padded, transcription_padded


class OfflineMultipleSpeakerMaskCollateFunc:
    def __init__(
        self,
        max_audio_length: int,
        mask_prob: Tuple[float],
        audio_mask_token: int,
        audio_pad_token: int,
        audio_eos_token: int,
        text_pad_token: int,
        mask_type: str,
        audio_pad_type: str,
        use_eos_as_pad: bool = False,
        loss_on_eos_pad: bool = False,
        pad_loss_weight: float = 1.0,
    ):
        """
        Initializes the collate function for masking audio features.

        Params:
            max_audio_length (int): The maximum length of the audio sequences.
            mask_prob (Tuple[float]): The probability range for masking audio tokens, (min, max).
            audio_mask_token (int): The token ID used for masking audio tokens.
            audio_pad_token (int): The token ID used for padding audio tokens.
            audio_eos_token (int): The token ID used for the end of audio sequences.
            text_pad_token (int): The token ID used for padding text tokens.
            mask_type (str): The type of masking to apply ("random" or "contiguous").
            audio_pad_type (str): The padding type for audio sequences ("variable" or "fixed").
            use_eos_as_pad (bool): Whether to use the EOS token as padding.
            loss_on_eos_pad (bool): Whether to compute loss on the EOS padding tokens.
            pad_loss_weight (float): The weight for the loss on padding tokens (between 0.0 and 1.0), should be between 0.0 and 1.0 and use_eos_as_pad and loss_on_eos_pad must be True.
        """
        self.max_audio_length = max_audio_length
        self.mask_prob = mask_prob

        self.audio_mask_token = audio_mask_token
        self.audio_pad_token = audio_pad_token
        self.audio_eos_token = audio_eos_token

        self.text_pad_token = text_pad_token

        self.mask_type = mask_type
        self.audio_pad_type = audio_pad_type

        self.use_eos_as_pad = use_eos_as_pad
        self.loss_on_eos_pad = loss_on_eos_pad

        assert 0.0 <= pad_loss_weight <= 1.0, f"Invalid pad_loss_weight: {pad_loss_weight}, should be between 0.0 and 1.0"
        self.pad_loss_weight = pad_loss_weight

        if self.use_eos_as_pad:
            print("\n\tUsing EOS token as padding!!!\n")
        if self.loss_on_eos_pad:
            print("\n\tComputing loss on EOS padding tokens!!!\n")
        if self.pad_loss_weight != 1.0:
            print(f"\n\tUsing pad loss weight: {self.pad_loss_weight}!!!\n")

    def time_scheduler_cubic_kappa(self, t: torch.Tensor, a: float = 0.0, b: float = 2.0) -> torch.Tensor:
        return (-2*t**3 + 3*t**2 + a*(t**3 - 2*t**2 + t) + b*(t**3 - t**2)).clamp(0.0, 1.0)

    def mask_audio_and_create_loss_mask(
        self,
        audio_codes_padded: torch.Tensor,
        audio_att_mask: torch.Tensor,  # NEW: True up to (len_with_eos), False after
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            masked_audio: same shape as input
            mask: bool [B, L], True = position contributes to loss (i.e., was masked)
        """
        B, L = audio_codes_padded.shape

        # Sample masking prob per sequence (on-device)
        conditioning_rev_rate = torch.empty(B, device=audio_codes_padded.device, dtype=torch.float32)
        conditioning_rev_rate.uniform_(self.mask_prob[0], self.mask_prob[1])
        # kappa_t = self.time_scheduler_cubic_kappa(time_step)
        masking_rate = 1.0 - conditioning_rev_rate
        # Decide where masking *may* happen.
        # - If using EOS-as-pad and training on the EOS tail, valid = all positions.
        # - Else valid = audio_att_mask (i.e., tokens up to and incl. the gold EOS).
        if self.use_eos_as_pad and self.loss_on_eos_pad:
            valid = torch.ones_like(audio_codes_padded, dtype=torch.bool)
        else:
            valid = audio_att_mask.bool()

        if self.mask_type == "random":
            rand = torch.rand(B, L, device=audio_codes_padded.device)
            mask = (rand < masking_rate[:, None]) & valid

            masked_audio = audio_codes_padded.clone()
            masked_audio[mask] = self.audio_mask_token

        elif self.mask_type == "contiguous":
            masked_audio = audio_codes_padded.clone()
            mask = torch.zeros_like(audio_codes_padded, dtype=torch.bool)

            for i in range(B):
                if self.use_eos_as_pad and self.loss_on_eos_pad:
                    # allow contiguous blocks *anywhere* (including EOS tail)
                    valid_length = L
                    valid_idx = torch.arange(L, device=audio_codes_padded.device)
                else:
                    # contiguous blocks only where audio_att_mask==True
                    valid_idx = audio_att_mask[i].nonzero(as_tuple=False).squeeze(-1)
                    valid_length = valid_idx.numel()

                if valid_length > 0:
                    block_length = max(1, int(valid_length * float(masking_rate[i].item())))
                    start_max = max(0, valid_length - block_length)
                    start = 0 if start_max == 0 else torch.randint(0, start_max + 1, (1,), device=audio_codes_padded.device).item()

                    if self.use_eos_as_pad and self.loss_on_eos_pad:
                        idx = torch.arange(start, start + block_length, device=audio_codes_padded.device)
                    else:
                        idx = valid_idx[start:start + block_length]

                    mask[i, idx] = True
                    masked_audio[i, idx] = self.audio_mask_token
        else:
            raise ValueError(f"Unknown mask_type: {self.mask_type}")

        return masked_audio, mask

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
        audio_codes, tokenized_transcription_list = zip(*batch)

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
        audio_att_mask = (arangeL < lengths_with_eos.unsqueeze(1))  # [B, L] bool

        # Masking (now uses audio_att_mask and respects the two new flags)
        masked_audio_codes, _ = self.mask_audio_and_create_loss_mask(audio_codes_padded, audio_att_mask)

        if self.use_eos_as_pad and self.loss_on_eos_pad and self.pad_loss_weight < 1.0:
            # 1.0 for content + gold EOS; pad positions get down-weighted
            loss_weight = torch.ones_like(audio_codes_padded, dtype=torch.float32)
            loss_weight[~audio_att_mask] = self.pad_loss_weight
        else:
            loss_weight = None

        # Text padding + attention mask (unchanged)
        transcription_padded = pad_sequence(
            tokenized_transcription_list,
            batch_first=True,
            padding_value=self.text_pad_token
        )
        transcription_attention_mask = (transcription_padded != self.text_pad_token)

        # x_1 = audio_codes_padded
        # x_t = masked_audio_codes
        # text_cond = transcription_padded
        # text_cond_mask = transcription_attention_mask
        # t = times_t
        return audio_codes_padded, transcription_padded, transcription_attention_mask, masked_audio_codes, audio_att_mask, loss_weight



class OfflineVoiceCloningSimplifiedCollateFunc:
    def __init__(
        self,
        max_audio_length: int,
        audio_pad_token: int,
        audio_eos_token: int,
        text_pad_token: int,
        audio_pad_type: str,
        use_eos_as_pad: bool = False,
    ):
        """
        Initializes the collate function for masking audio features.

        Params:
            max_audio_length (int): The maximum length of the audio sequences.
            audio_pad_token (int): The token ID used for padding audio tokens.
            audio_eos_token (int): The token ID used for the end of audio sequences.
            text_pad_token (int): The token ID used for padding text tokens.
            mask_type (str): The type of masking to apply ("random" or "contiguous").
            audio_pad_type (str): The padding type for audio sequences ("variable" or "fixed").
            use_eos_as_pad (bool): Whether to use the EOS token as padding.
        """
        self.max_audio_length = max_audio_length

        self.audio_pad_token = audio_pad_token
        self.audio_eos_token = audio_eos_token

        self.text_pad_token = text_pad_token
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
        audio_codes, tokenized_transcription_list = zip(*batch)

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

        # Text padding + attention mask (unchanged)
        transcription_padded = pad_sequence(
            tokenized_transcription_list,
            batch_first=True,
            padding_value=self.text_pad_token
        )
        transcription_att_mask = (transcription_padded != self.text_pad_token)

        x_1 = audio_codes_padded
        x_1_att_mask = audio_att_mask

        return x_1, x_1_att_mask, transcription_padded, transcription_att_mask


class DurationBPEOfflineDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        base_dir: str,
        filepath_column: str,
        text_tokenizer: VoiceBpeTokenizer,
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

        if "language" in datum:
            language = datum["language"]
        else:
            language = "en"
        # try:
        audio_codes = self._load_codes(filename)
        tokenized_transcription = torch.tensor(
            self.text_tokenizer.encode(
                transcription,
                lang=language
            )
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


class OfflineMultipleSpeakerDreamOnCollateFunc:
    def __init__(
        self,
        max_audio_length: int = 2048,
        audio_mask_token: int = None,
        audio_pad_token: int = None,
        audio_expand_token: int = None,
        audio_eos_token: int = None,
        text_pad_token: int = 0,
        mask_type: str = "contiguous",
        mask_prob: Tuple[float, float] = (0.7, 1.0),
        audio_pad_type: str = "variable",      # "variable" or "fixed"
        # DreamOn params:
        mix_ratio: float = 0.5,
        p_merge_static: float = 0.25,
        p_merge_dynamic_scale: float = 0.5,
        delete_frac_range: tuple = (0.0, 0.10),  # how many EOS-in-middle to insert (fraction of current length)
        delete_loss_weight: float = 0.5,
        # padding params
        use_eos_as_pad: bool = False,
        loss_on_eos_pad: bool = False,
        pad_loss_weight: float = 1.0,
    ):
        self.max_audio_length = max_audio_length
        self.audio_mask_token = audio_mask_token
        self.audio_pad_token = audio_pad_token
        self.audio_expand_token = audio_expand_token
        self.audio_eos_token = audio_eos_token
        self.text_pad_token = text_pad_token
        self.mask_type = mask_type
        self.mask_prob = mask_prob
        self.audio_pad_type = audio_pad_type

        self.mix_ratio = mix_ratio
        self.p_merge_static = p_merge_static
        self.p_merge_dynamic_scale = p_merge_dynamic_scale
        self.delete_frac_range = delete_frac_range
        self.delete_loss_weight = delete_loss_weight

        self.use_eos_as_pad = use_eos_as_pad
        self.loss_on_eos_pad = loss_on_eos_pad
        assert 0.0 <= pad_loss_weight <= 1.0, f"Invalid pad_loss_weight: {pad_loss_weight}"
        self.pad_loss_weight = pad_loss_weight
        if self.use_eos_as_pad:
            print("\n\tUsing EOS token as padding!!!\n")
        if self.loss_on_eos_pad:
            print("\n\tComputing loss on EOS padding tokens!!!\n")
        if self.pad_loss_weight != 1.0:
            print(f"\n\tUsing pad loss weight: {self.pad_loss_weight}!!!\n")

    def _pad_1d(self, x: torch.Tensor, L: int, value: int) -> torch.Tensor:
        out = x.new_full((L,), value)
        Lx = min(L, x.numel())
        out[:Lx] = x[:Lx]
        return out

    def _append_eos_then_pad(self, x: torch.Tensor, L: int) -> torch.Tensor:
        """Make room for EOS if needed, append EOS, then pad with PAD to L."""
        x = x.view(-1).long()
        if L <= 0:
            return x.new_zeros((0,), dtype=torch.long)
        # leave room for EOS if we must truncate
        core_max = max(0, L - 1)
        x = x[:core_max]
        x = torch.cat([x, x.new_tensor([self.audio_eos_token])], dim=0)
        if x.numel() < L:
            # x = torch.cat([x, x.new_full((L - x.numel(),), self.audio_pad_token)], dim=0)
            effective_pad_id = self.audio_eos_token if self.use_eos_as_pad else self.audio_pad_token
            x = torch.cat([x, x.new_full((L - x.numel(),), effective_pad_id)], dim=0)
        return x

    def _find_spans(self, mask_bool: torch.Tensor):
        spans = []
        if mask_bool.numel() == 0:
            return spans
        in_run = False
        start = 0
        for i, v in enumerate(mask_bool.tolist() + [False]):
            if v and not in_run:
                in_run = True
                start = i
            elif not v and in_run:
                in_run = False
                spans.append((start, i))  # [start, end)
        return spans

    def _mask_audio_random_or_contiguous(self, audio: torch.Tensor, audio_att_mask: torch.Tensor):
        """
        Per-sequence masking (each sequence samples its own p in [min,max]).
        Returns a boolean mask of which positions are masked.
        """
        L = audio.numel()
        mask_bool = torch.zeros(L, dtype=torch.bool, device=audio.device)

        p_min, p_max = self.mask_prob
        p = torch.empty(1, device=audio.device).uniform_(p_min, p_max).item()

        if self.use_eos_as_pad and self.loss_on_eos_pad:
            valid = torch.ones(L, dtype=torch.bool, device=audio.device)
        else:
            valid = audio_att_mask.bool()

        if self.mask_type == "contiguous":
            valid_idx = valid.nonzero(as_tuple=False).squeeze(-1)
            if valid.numel() > 0:
                valid_L = valid.numel()
                block_len = max(1, int(valid_L * p))
                start = torch.randint(0, max(1, valid_L - block_len + 1), (1,), device=audio.device).item()
                idx = valid_idx[start : start + block_len]
                mask_bool[idx] = True
        else:
            rand = torch.rand_like(audio, dtype=torch.float)
            mask_bool = (rand < p) & valid

        return mask_bool

    def _merge_masks_into_expand(self, src_tokens: list, mask_bool: torch.Tensor):
        """
        Pairwise merge masked tokens to <EXPAND> with probability p.
        Leftover singles remain masked (to be masked in input).
        """
        spans = self._find_spans(mask_bool)
        num_mask = int(mask_bool.sum().item())
        p_dyn = 0.0 if num_mask == 0 else min(1.0, self.p_merge_dynamic_scale / float(num_mask))
        p = self.mix_ratio * self.p_merge_static + (1.0 - self.mix_ratio) * p_dyn

        out_tokens, out_mask = [], []
        i = 0
        for (s, e) in spans:
            # copy region before span
            while i < s:
                out_tokens.append(src_tokens[i]); out_mask.append(False); i += 1
            # inside span
            span_len = e - s
            j = 0
            while j < span_len:
                if (j + 1) < span_len and torch.rand(1).item() < p:
                    out_tokens.append(self.audio_expand_token)
                    out_mask.append(True)   # EXPAND is supervised (masked in input)
                    j += 2
                else:
                    out_tokens.append(src_tokens[s + j])
                    out_mask.append(True)   # still masked
                    j += 1
            i = e
        # tail
        while i < len(src_tokens):
            out_tokens.append(src_tokens[i]); out_mask.append(False); i += 1
        return out_tokens, out_mask

    def _insert_middle_eos(self, z0_tokens: list, z0_mask: list):
        """
        Insert EOS *inside* the editable region (not the terminal EOS),
        to teach delete semantics via EOS-in-the-middle. The inserted EOS positions
        are supervised (masked in input). Count ~ Uniform(delete_frac_range) * L.
        """
        L = len(z0_tokens)
        if L == 0:
            return z0_tokens, z0_mask

        lo, hi = self.delete_frac_range
        k = int(torch.empty(1).uniform_(lo, hi).item() * L)
        if k <= 0:
            return z0_tokens, z0_mask

        # avoid placing EOS right after sentinels or on PAD/MASK/EXPAND/EOS
        SENTINELS = {
            self.audio_mask_token,
            self.audio_expand_token,
            self.audio_eos_token,
        }

        eligible = []
        for t in range(1, L - 1):  # keep last EOS (appended) intact; need a left neighbor
            left = z0_tokens[t - 1]
            cur  = z0_tokens[t]
            if left not in SENTINELS and cur not in SENTINELS:
                eligible.append(t)

        if len(eligible) == 0:
            return z0_tokens, z0_mask

        k = min(k, len(eligible))
        chosen = sorted(torch.randperm(len(eligible))[:k].tolist())
        ins_positions = [eligible[idx] for idx in chosen]

        out_t, out_m = [], []
        src_i = 0
        ins_i = 0
        for t in range(L + k):
            # We compare against src index + how many we've already inserted before it
            if ins_i < k and (src_i + ins_i) == ins_positions[ins_i]:
                out_t.append(self.audio_eos_token)  # middle EOS (delete signal)
                out_m.append(True)                  # supervised → masked in input
                ins_i += 1
            else:
                out_t.append(z0_tokens[src_i])
                out_m.append(z0_mask[src_i])
                src_i += 1

        return out_t, out_m

    def _build_input_from_z0(self, z0_tokens: list, z0_mask: list):
        """
        Build model input by masking only the supervised positions and EXPAND sentinels.
        - Terminal EOS has z0_mask=False → stays visible.
        - Middle EOS has z0_mask=True   → becomes MASK (supervised).
        """
        inp = []
        loss_mask = []
        for tok, m in zip(z0_tokens, z0_mask):
            is_expand = (tok == self.audio_expand_token)
            use_mask = m or is_expand
            inp.append(self.audio_mask_token if use_mask else tok)
            loss_mask.append(bool(use_mask))
        return inp, torch.tensor(loss_mask, dtype=torch.bool)

    def __call__(self, batch):
        # unpack
        audio_codes, tokenized_transcription_list = zip(*batch)

        # choose target length (include room for EOS)
        if self.audio_pad_type == "variable":
            # +1 for EOS on each sequence, then cap by max_audio_length
            lengths_plus_eos = [int(a.view(-1).numel()) + 1 for a in audio_codes]
            max_L = min(max(lengths_plus_eos), self.max_audio_length)
        elif self.audio_pad_type == "fixed":
            max_L = self.max_audio_length
        else:
            raise ValueError(f"Unknown audio_pad_type: {self.audio_pad_type}")

        z0_list, inp_list, m_list = [], [], []
        lengths_with_eos = []

        for audio in audio_codes:
            a = audio.clone().long().view(-1)
            # length used to build attn mask later (len content + EOS)
            core_len = min(max_L - 1, a.numel())
            lengths_with_eos.append(core_len + 1)

            # (0) append EOS then PAD to max_L
            a = self._append_eos_then_pad(a, max_L)              # shape [max_L]

            # Build per-seq audio_att_mask: True up to content+EOS, False after
            arangeL = torch.arange(max_L, device=a.device)
            audio_att_mask_i = (arangeL < (core_len + 1))  # bool

            # (1) pick mask set (per-sequence probability)
            mask_bool = self._mask_audio_random_or_contiguous(a, audio_att_mask_i)

            # (2) z0: start from original tokens (a), merge masked spans -> <EXPAND>
            z0_tokens, z0_mask = self._merge_masks_into_expand(a.tolist(), mask_bool)

            # (3) insert EOS *in the middle* as delete signals (supervised)
            z0_tokens, z0_mask = self._insert_middle_eos(z0_tokens, z0_mask)

            # (4) input to model & loss mask
            inp_tokens, loss_mask = self._build_input_from_z0(z0_tokens, z0_mask)

            # (5) truncate (safety) to max_L and pad together
            Lcap = min(max_L, len(z0_tokens))
            z0_tokens  = z0_tokens[:Lcap]
            inp_tokens = inp_tokens[:Lcap]
            loss_mask  = loss_mask[:Lcap]

            effective_pad_id = self.audio_eos_token if self.use_eos_as_pad else self.audio_pad_token
            z0_list.append(self._pad_1d(torch.as_tensor(z0_tokens, dtype=torch.long, device=a.device), max_L, effective_pad_id))
            inp_list.append(self._pad_1d(torch.as_tensor(inp_tokens, dtype=torch.long, device=a.device), max_L, effective_pad_id))
            m_list.append(self._pad_1d(loss_mask.to(torch.long), max_L, 0).bool())

        z0_padded    = torch.stack(z0_list, dim=0)
        input_padded = torch.stack(inp_list, dim=0)
        loss_mask    = torch.stack(m_list,  dim=0)

        # --- Batch audio attention mask (for loss weighting like MaskCollate) ---
        lengths_with_eos_t = torch.tensor(lengths_with_eos, device=z0_padded.device, dtype=torch.long)
        arangeL = torch.arange(max_L, device=z0_padded.device).unsqueeze(0)  # [1, L]
        audio_att_mask = (arangeL < lengths_with_eos_t.unsqueeze(1))         # [B, L] bool

        if self.use_eos_as_pad and self.loss_on_eos_pad and self.pad_loss_weight < 1.0:
            loss_weight = torch.ones_like(z0_padded, dtype=torch.float32)
            loss_weight[~audio_att_mask] = self.pad_loss_weight
        else:
            loss_weight = None

        # --- Text padding + attention mask ---
        transcription_padded = pad_sequence(
            tokenized_transcription_list,
            batch_first=True,
            padding_value=self.text_pad_token
        )
        transcription_attention_mask = (transcription_padded != self.text_pad_token)  # bool

        return z0_padded, transcription_padded, transcription_attention_mask, input_padded, loss_mask, loss_weight


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
    """

    def __init__(
        self,
        text_tokenizer: AutoTokenizer,
        text_column: str,
        audio_column: str = "audio",
        sampling_rate: int = 16_000,
        max_audio_duration: Optional[float] = None,
    ):
        self.text_tokenizer = text_tokenizer
        self.text_column = text_column
        self.audio_column = audio_column
        self.sampling_rate = sampling_rate
        self.max_audio_frames = (
            int(max_audio_duration * sampling_rate)
            if max_audio_duration is not None
            else None
        )

    def _prepare_waveform(self, audio) -> torch.Tensor:
        waveform = torch.tensor(audio["array"], dtype=torch.float32)
        source_sr = int(audio["sampling_rate"])

        # Hugging Face Audio is normally mono here, but keep stereo robust.
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)
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

            waveforms.append(
                self._prepare_waveform(sample[self.audio_column])
            )
            transcriptions.append(str(transcription))

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



@torch.no_grad()
def main():
    metadata_path = "/raid/aluno_alef/DATASETS/train_dfm_ablation.csv"
    data = pd.read_csv(metadata_path)

    if "language" not in data.columns:
        data["language"] = "en"

    # text_tokenizer = VoiceBpeTokenizer(vocab_file="../config/vocab.json")

    # print(text_tokenizer.tokenizer)
    # print(text_tokenizer.tokenizer.get_vocab()["[START]"])

    # dataset = OfflineMultipleSpeakerDataset(
    #     data=data,
    #     base_dir="",
    #     filepath_column="codec_filepath",
    #     text_tokenizer=text_tokenizer
    # )

    # collate_fn = OfflineMultipleSpeakerMaskCollateFunc(
    #     max_audio_length=2048,
    #     mask_prob=(0.7, 1.0),
    #     audio_mask_token=65536,
    #     audio_pad_token=65537,
    #     text_pad_token=0,
    #     mask_type="contiguous",
    #     # mask_type="random",
    #     audio_pad_type="variable", # can be "variable" or "fixed"
    #     # audio_pad_type="fixed", # can be "variable" or "fixed"
    # )

    # collate_fn = OfflineMultipleSpeakerDreamOnCollateFunc(
    #     max_audio_length=2048,
    #     mask_prob=(0.7, 1.0),
    #     audio_mask_token=65536,
    #     audio_pad_token=65537,
    #     text_pad_token=0,
    #     # mask_type="contiguous",
    #     mask_type="random",
    #     audio_pad_type="variable", # can be "variable" or "fixed"
    #     # audio_pad_type="fixed", # can be "variable" or "fixed"
    #     audio_expand_token=65538,
    #     audio_delete_token=65539,
    #     mix_ratio=0.5,
    #     p_merge_static=0.25,
    #     p_merge_dynamic_scale=0.5,
    #     delete_frac_range=(0.0, 0.10),
    #     delete_loss_weight=0.5,
    # )

    text_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    # add pad token if not present, make it equal to eos token
    if text_tokenizer.pad_token is None:
        text_tokenizer.add_special_tokens({'pad_token': text_tokenizer.eos_token})

    # print vocab size
    print(f"Text tokenizer vocab size: {text_tokenizer.vocab_size}")
    print(f"Text tokenizer pad_token: {text_tokenizer.pad_token_id}")

    dataset = HFTextTokenizerDataset(
        data=data,
        base_dir="",
        filepath_column="codec_filepath",
    )

    collate_fn = HFTextTokenizerCollator(
        text_tokenizer=text_tokenizer,
        max_audio_length=2048,
        audio_pad_token=65537,
        audio_eos_token=65536,
        audio_pad_type="variable",
        use_eos_as_pad=False,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    for x1, x1_att, transcription_ids, transcription_ids_att in dataloader:
        print(x1.shape)
        print(transcription_ids.shape)
        print(transcription_ids_att.shape)

        print(x1)
        print(x1_att)

        print(transcription_ids)
        print(transcription_ids_att)

        # print(cond)
        # print("="*100)
        # print(x_1)
        # print(transcription_ids)
        break

    # print()

    # metadata_path = "/raid/aluno_alef/DATASETS/xcodec2/LibriTTS_R/libri_tts-train-clean-960.csv"
    # data = pd.read_csv(metadata_path)

    # text_tokenizer = VoiceBpeTokenizer(vocab_file="../config/vocab.json")

    # dataset = DurationBPEOfflineDataset(
    #     data=data,
    #     base_dir="/raid/aluno_alef/DATASETS/xcodec2/LibriTTS_R",
    #     text_tokenizer=text_tokenizer
    # )

    # collate_fn = DurationBPEOfflineCollateFunc(
    #     max_audio_length=2048,
    #     text_pad_token=0,
    #     audio_pad_token=65537,
    #     audio_bos_token=65536,  # BOS token for audio
    # )

    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=4,
    #     shuffle=True,
    #     num_workers=8,
    #     collate_fn=collate_fn
    # )

    # for batch in dataloader:
    #     print(batch)
    #     break





if __name__ == "__main__":
    main()