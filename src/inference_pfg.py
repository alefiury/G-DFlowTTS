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
from torch.distributions.categorical import Categorical
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

@torch.inference_mode()
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
    xt = source_distribution.sample((1, max_length), device=device)

    print("-"*100)
    print(xt.shape, codes_ref.shape)

    # x_t[:, : codes_ref.shape[-1]] = codes_ref

    if codes_ref.size(0) < config.datasets.max_audio_length:
        codes_ref = F.pad(codes_ref, (0, config.datasets.max_audio_length - codes_ref.size(0)), value=config.datasets.audio_mask_token).unsqueeze(0)

    print(codes_ref)
    print(codes_ref.shape)

    num_steps = nsf
    dt = 1.0 / num_steps
    x1_temp = 1.0
    guidance_scale = config.datasets.guidance_scale
    # gamma = config.datasets.guidance_scale
    gamma = 5
    mask_token_id = config.datasets.audio_mask_token
    S = vocab_size
    eps = 1e-9
    noise = 0

    mask_one_hot = torch.zeros((S), device=model.device)
    mask_one_hot[mask_token_id] = 1.0

    # Loop over the time grid
    for step in range(num_steps):
        t_val    = step * dt
        t_tensor = xt.new_full((1,), t_val)

        # unconditional pass
        logits_u = model(xt, text_ids, codes_ref, t_tensor, True,  True)
        probs_u  = torch.softmax(logits_u / x1_temp, -1)

        # conditional pass
        logits_c = model(xt, text_ids, codes_ref, t_tensor, False, False)
        probs_c  = torch.softmax(logits_c / x1_temp, -1)

        xt_mask  = (xt == mask_token_id).unsqueeze(-1).float()
        base_r   = (1 + noise * t_val) / (1 - t_val)

        R_u = xt_mask * probs_u * base_r
        R_c = xt_mask * probs_c * base_r

        remask = (1 - xt_mask) * mask_one_hot.view(1,1,S) * noise
        R_u += remask;  R_c += remask

        log_Ru = torch.log(R_u + eps)
        log_Rc = torch.log(R_c + eps)
        R_mix  = torch.exp(gamma * log_Rc + (1 - gamma) * log_Ru)

        # enforce row‑sum zero
        R_mix.scatter_(-1, xt[..., None], 0.)
        R_mix.scatter_(-1, xt[..., None], -R_mix.sum(-1, keepdim=True))

        # Euler step
        P = (R_mix * dt).clamp_min(0.)
        diag = (1. - P.sum(-1, keepdim=True)).clamp_min(0.)
        P.scatter_(-1, xt[..., None], diag)

        xt = torch.multinomial(P.view(-1, S), 1).view_as(xt)

    return xt

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

    os.makedirs("outputs_pfg", exist_ok=True)

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
            torchaudio.save(f"outputs_pfg/audio_{idx}-{n}.wav", generated_audio.squeeze(0).cpu(), 16000)


if __name__ == "__main__":
    main()
