import os
import json
import logging
import argparse
import warnings
from pprint import pprint
from typing import Tuple
warnings.filterwarnings("ignore")

import wandb
import torch

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
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from modules.pl_wrapper import DFMTTSWrapper
from utils.tokenizer import VoiceBpeTokenizer


class MaskedSourceDistribution():
    def __init__(self, mask_token: int) -> None:
        self.mask_token = mask_token

    @property
    def masked(self) -> bool:
        return True

    def sample(self, tensor_size: Tuple[int, ...], device: torch.device) -> Tensor:
        return torch.zeros(tensor_size, device=device).fill_(self.mask_token).long()

    def sample_like(self, tensor_like: Tensor) -> Tensor:
        return torch.zeros_like(tensor_like).fill_(self.mask_token).long()

@torch.no_grad()
def inference(config, model, tokenizer, sentence, nsf: int = 10, codes_ref: Tensor = None, device: torch.device = torch.device("cuda")) -> Tensor:
    text_ref = "in being comparatively modern."
    augmented_sentence = text_ref + " " + sentence
    text_ids = tokenizer.encode(augmented_sentence, lang="en-us")

    text_ids = torch.tensor(text_ids).unsqueeze(0).to(device)
    max_length = config.datasets.max_audio_length
    vocab_size = config.datasets.audio_vocab_size + config.model.add_token

    source_distribution = MaskedSourceDistribution(
        mask_token=config.datasets.audio_mask_token
    )
    # Set the number of predictor steps (you can adjust this or read it from config)
    num_steps = nsf  # for example, 10 steps from t=0 to t=1
    num_corrector_steps = 1  # number of corrector iterations per predictor step

    # Create a time grid from 0 to 1 with (num_steps + 1) points
    t_init = 0.0
    t_final = 1.0
    time_grid = torch.linspace(t_init, t_final, num_steps + 1, device=device)

    # Initialize x_t; for example, using the masked source
    x_t = source_distribution.sample((1, max_length), device=device)

    print("-"*100)
    print(x_t.shape, codes_ref.shape)

    # x_t[:, : codes_ref.shape[-1]] = codes_ref

    if codes_ref.size(0) < config.datasets.max_audio_length:
        codes_ref = F.pad(codes_ref, (0, config.datasets.max_audio_length - codes_ref.size(0)), value=config.datasets.audio_mask_token).unsqueeze(0)

    print(codes_ref)
    print(codes_ref.shape)

    guidance_scale = 8

    # Loop over the time grid
    for i in tqdm(range(num_steps), total=num_steps):
        t = time_grid[i : i + 1]         # current time, shape [1]
        h = time_grid[i + 1] - time_grid[i]  # step size (scalar)
        # Predictor update: compute model output and update x_t
        logits_c = model(
            x_t=x_t,
            text_ids=text_ids,
            cond_ids=codes_ref,
            time=t,
            drop_text=False,
            drop_cond=False,
        )
        logits_u = model(
            x_t=x_t,
            text_ids=text_ids,
            cond_ids=codes_ref,
            time=t,
            drop_text=True,
            drop_cond=True,
        )
        logits = logits_c + guidance_scale * (logits_c - logits_u)
        p1 = torch.softmax(logits, dim=-1)
        one_hot_x_t = torch.nn.functional.one_hot(x_t, num_classes=vocab_size).float()

        # Compute the velocity update using the denoiser formulation
        # Here, u = (p1 - one_hot_x_t) / (1 - t), note the small epsilon for numerical stability.
        u = (p1 - one_hot_x_t) / (1.0 - t.item() + 1e-8)

        # Euler update: compute new probabilities and sample the updated state
        new_probs = one_hot_x_t + h * u
        new_probs = new_probs / new_probs.sum(dim=-1, keepdim=True)
        x_t = torch.distributions.Categorical(probs=new_probs).sample()

        # # Optional: Corrector iterations at the current time step
        # for _ in range(num_corrector_steps):
        #     # Use a smaller corrector step (for example, 10% of h)
        #     h_corr = h * 0.1
        #     logits_corr = model(
        #         x_t=x_t,
        #         text_ids=text_ids,
        #         cond_ids=codes_ref,
        #         time=t,
        #         drop_text=False,
        #         drop_cond=False,
        #     )
        #     p1_corr = torch.softmax(logits_corr, dim=-1)
        #     one_hot_x_t_corr = torch.nn.functional.one_hot(x_t, num_classes=vocab_size).float()

        #     # Compute the corrector velocity similarly
        #     u_corr = (p1_corr - one_hot_x_t_corr) / (1.0 - t.item() + 1e-8)
        #     new_probs_corr = one_hot_x_t_corr + h_corr * u_corr
        #     new_probs_corr = new_probs_corr / new_probs_corr.sum(dim=-1, keepdim=True)
        #     x_t = torch.distributions.Categorical(probs=new_probs_corr).sample()

    return x_t

@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_path",
        required=True,
        type=str,
        help="YAML file with configurations"
    )
    parser.add_argument(
        "-g",
        "--gpu",
        default=0,
        required=False,
        type=int
    )
    parser.add_argument(
        "-pc",
        "--pretrained-checkpoint",
        required=False,
        type=str,
        default=None
    )

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    tokenizer = VoiceBpeTokenizer(vocab_file=config.datasets.vocab_file)
    model = DFMTTSWrapper.load_from_checkpoint(args.pretrained_checkpoint, config=config, map_location=device, strict=False)
    model.eval()

    audio_codec = XCodec2Model.from_pretrained(config.datasets.audio_codec).to(device)
    audio_codec.eval()

    ref_path = "/hadatasets/alef.ferreira/DFM-TTS-2/src/samples/LJ001-0002.wav"

    audio_ref, audio_ref_sr = torchaudio.load(ref_path)
    if audio_ref_sr != 16000:
        audio_ref = torchaudio.transforms.Resample(audio_ref_sr, 16000)(audio_ref)

    print(f"Audio reference shape: {audio_ref.shape}")

    codes_ref = audio_codec.encode_code(input_waveform=audio_ref).squeeze()

    sentences = [
        "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition",
        "For although the Chinese took impressions from wood blocks engraved in relief for centuries before the woodcutters of the Netherlands, by a similar process",
        "Hello, how are you?",
        "The quick brown fox jumps over the lazy dog.",
        "The five boxing wizards jump quickly.",
        "How razorback-jumping frogs can level six piqued gymnasts!",
        "Pack my box with five dozen liquor jugs."
    ]

    nsf = [128, 256, 512, 1024, 2048]

    os.makedirs("outputs_2_pfg", exist_ok=True)

    for idx, sentence in enumerate(sentences):
        print(f"Processing sentence {idx + 1}/{len(sentences)}")
        for n in tqdm(nsf):
            x_t = inference(config, model, tokenizer, sentence, nsf=n, codes_ref=codes_ref, device=device)
            # remove making tokens from the generated sequence
            x_t = x_t.squeeze(0)
            print("1", x_t.shape)
            x_t = x_t[x_t != config.datasets.audio_mask_token]
            print("2", x_t.shape)
            # remove padding tokens from the generated sequence
            x_t = x_t[x_t != config.datasets.audio_pad_token]
            print("3", x_t.shape)
            print(x_t)
            x_t = x_t.unsqueeze(0).unsqueeze(0)
            print("4", x_t.shape)
            # Decode the final token sequence into an audio waveform
            generated_audio = audio_codec.decode_code(x_t)
            print("5", generated_audio.shape)
            torchaudio.save(f"outputs_2_pfg/audio_{idx}-{n}.wav", generated_audio.squeeze(0).cpu(), 16000)


if __name__ == "__main__":
    main()
