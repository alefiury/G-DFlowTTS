import os
import json
import logging
import argparse
import warnings
import shutil
from pprint import pprint
from typing import Tuple, Union
warnings.filterwarnings("ignore")

import wandb
import torch
import argparse

import pandas as pd
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from torch import Tensor
import torchaudio
from omegaconf import OmegaConf
from lightning.pytorch import Trainer
from transformers import AutoTokenizer
from xcodec2.modeling_xcodec2 import XCodec2Model
from lightning.pytorch.loggers import WandbLogger
from torch.distributions.categorical import Categorical
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from torchaudio.transforms import Resample
from neucodec import NeuCodec

from modules.pl_wrapper import DFMTTSWrapper
from modules.dp_wrapper import DurationPredictorWrapper
from utils.tokenizer import VoiceBpeTokenizer


def _ensure_mono_16k(wav_path: str, sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(wav_path)
    """(C,T) -> (1,T_16k) as float32 in [-1,1]."""
    if wav.dim() != 2:
        raise ValueError(f"Expected waveform shape (C,T), got {tuple(wav.shape)}")
    if wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != 16_000:
        wav = Resample(sr, 16_000)(wav)
    # Clamp to [-1,1] just in case
    wav = wav.clamp_(-1.0, 1.0)
    return wav


@torch.inference_mode()
def _encode_audio(model, wav_path: str) -> torch.Tensor:
    """
    Try common encode methods to obtain integer code sequence.
    Returns a 1-D LongTensor of shape (L,)
    """
    wav_1c16k = _ensure_mono_16k(wav_path, 16_000)
    # print(wav_1c16k[None, ...].shape)
    out = model.encode_code(wav_1c16k[None, ...])
    # Squeeze batch and channel dims
    return out.long().cpu()


@torch.no_grad()
def main():
    output_dir = "neucodec_generated_audios"
    libri_speech_test_clean_metadata = "/raid/aluno_alef/DATASETS/LibriSpeech-test-clean-filtered.csv"
    df = pd.read_csv(libri_speech_test_clean_metadata)

    print(df.columns)
    gpu = 0
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    audio_codec = NeuCodec.from_pretrained("neuphonic/neucodec").to(device)
    saving_sr = 24000
    audio_codec.eval()

    os.makedirs(output_dir, exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        gen_filename = f"gen_{idx}.wav"
        output_filepath = os.path.join(output_dir, gen_filename)
        codes_ref = _encode_audio(audio_codec, row["reference"]).to(device)
        print(f"Decoding codes of shape: {codes_ref.shape}")
        generated_audio = audio_codec.decode_code(codes_ref)

        shutil.copy(row["reference"], os.path.join(output_dir, f"ref_{idx}.wav"))
        torchaudio.save(output_filepath, generated_audio.squeeze(0).cpu(), saving_sr)


if __name__ == "__main__":
    main()
