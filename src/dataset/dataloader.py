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


from utils.tokenizer import VoiceBpeTokenizer
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
        max_audio_length: int = 2048,
        mask_prob: Tuple[float] = (0.7, 1.0),
        audio_mask_token: int = 0,
        audio_pad_token: int = 0,
        text_pad_token: int = 0,
        mask_type: str = "contiguous",
        audio_pad_type: str = "variable", # can be "variable" or "fixed"
    ):
        self.max_audio_length = max_audio_length
        self.mask_prob = mask_prob
        self.audio_mask_token = audio_mask_token
        self.audio_pad_token = audio_pad_token
        self.text_pad_token = text_pad_token

        self.mask_type = mask_type
        self.audio_pad_type = audio_pad_type

    def mask_audio_and_create_loss_mask(
        self,
        audio_codes_padded: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies masking to the audio codes and creates a loss mask.

        The masked audio codes replace tokens (except padding tokens) with mask_token
        according to the specified strategy. The loss mask indicates which tokens were masked
        (and are not padding) and therefore will be used for the loss calculation.

        Returns:
            masked_audio (Tensor): The audio codes after applying masking.
            loss_mask (Tensor): A boolean tensor with True for tokens to use in the loss.
        """
        # Sample mask_prob from a uniform distribution
        mask_prob = torch.rand(1).float().uniform_(*self.mask_prob).item()

        if self.mask_type == "random":
            # Only valid (non-padding) tokens are candidates.
            valid = (audio_codes_padded != self.audio_pad_token)
            rand = torch.rand(audio_codes_padded.shape, device=audio_codes_padded.device)
            mask = (rand < mask_prob) & valid
            masked_audio = audio_codes_padded.clone()
            masked_audio[mask] = self.audio_mask_token
        elif self.mask_type == "contiguous":
            masked_audio = audio_codes_padded.clone()
            B, L = audio_codes_padded.shape
            mask = torch.zeros_like(audio_codes_padded, dtype=torch.bool)
            for i in range(B):
                valid_idx = (audio_codes_padded[i] != self.audio_pad_token).nonzero(as_tuple=False).squeeze()
                if valid_idx.numel() > 0:
                    valid_length = valid_idx.numel()
                    block_length = max(1, int(valid_length * mask_prob))
                    if valid_length - block_length > 0:
                        start_idx = torch.randint(0, valid_length - block_length + 1, (1,)).item()
                    else:
                        start_idx = 0
                    indices_to_mask = valid_idx[start_idx:start_idx+block_length]
                    mask[i, indices_to_mask] = True
                    masked_audio[i, indices_to_mask] = self.audio_mask_token
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

        # Variable length padding
        if self.audio_pad_type == "variable":
            max_audio_length = max([audio.shape[-1] for audio in audio_codes])
            if max_audio_length > self.max_audio_length:
                max_audio_length = self.max_audio_length
        # Fixed length padding
        elif self.audio_pad_type == "fixed":
            max_audio_length = self.max_audio_length
        else:
            raise ValueError(f"Unknown audio_pad_type: {self.audio_pad_type}")

        # Audio Padding
        padded_audio_list = []
        for audio in audio_codes:
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)
            # pad the sample (assuming shape (1, L) or (L, D) if already multi-dimensional).
            padded_audio = self.pad_audio_codec(
                audio,
                max_audio_length,
                self.audio_pad_token
            )
            if padded_audio.size(0) == 1:
                padded_audio = padded_audio.squeeze(0)
            padded_audio_list.append(padded_audio)

        audio_codes_padded = torch.stack(padded_audio_list, dim=0)
        masked_audio_codes, mask = self.mask_audio_and_create_loss_mask(audio_codes_padded)
        # Text padding
        transcription_padded = pad_sequence(
            tokenized_transcription_list,
            batch_first=True,
            padding_value=self.text_pad_token
        )

        return audio_codes_padded, transcription_padded, masked_audio_codes, mask


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


@torch.no_grad()
def main():
    # metadata_path = "/hadatasets/alef.ferreira/DATASETS/LibriTTS_R/libri_tts-train-clean-460.csv"
    # data = pd.read_csv(metadata_path)

    # text_tokenizer = VoiceBpeTokenizer(vocab_file="../config/vocab.json")

    # dataset = OfflineMultipleSpeakerDataset(
    #     data=data,
    #     base_dir="/hadatasets/alef.ferreira/DATASETS/LibriTTS_R_xcodec2",
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

    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=4,
    #     shuffle=True,
    #     num_workers=8,
    #     collate_fn=collate_fn
    # )

    # for x_1, transcription_ids, cond in dataloader:
    #     print(x_1.shape)
    #     print(transcription_ids.shape)
    #     print(cond.shape)

    #     print(x_1)
    #     print(cond)
    #     print(transcription_ids)
    #     break

    metadata_path = "/raid/aluno_alef/DATASETS/xcodec2/LibriTTS_R/libri_tts-train-clean-960.csv"
    data = pd.read_csv(metadata_path)

    text_tokenizer = VoiceBpeTokenizer(vocab_file="../config/vocab.json")

    dataset = DurationBPEOfflineDataset(
        data=data,
        base_dir="/raid/aluno_alef/DATASETS/xcodec2/LibriTTS_R",
        text_tokenizer=text_tokenizer
    )

    collate_fn = DurationBPEOfflineCollateFunc(
        max_audio_length=2048,
        text_pad_token=0,
        audio_pad_token=65537,
        audio_bos_token=65536,  # BOS token for audio
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=8,
        collate_fn=collate_fn
    )

    for batch in dataloader:
        print(batch)
        break


if __name__ == "__main__":
    main()