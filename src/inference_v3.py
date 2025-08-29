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
from modules.dp_wrapper import DurationPredictorWrapper
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


def get_remaining_duration(
    duration_model: DurationPredictorWrapper,
    text_ids: Tensor,
    codes_ref: Tensor,
    device: torch.device
) -> Tensor:
    bos_vec = codes_ref.new_full((1,), 65536)
    codes_ref = torch.cat((bos_vec, codes_ref), dim=0)   # [C, dur+1]

    remaining_duration = duration_model(
        text_ids=text_ids,
        audio_ids=codes_ref.unsqueeze(0).to(device)
    )
    return torch.argmax(remaining_duration[:, -1], dim=-1).item()

# >>> NEW: helper to apply temperature / top-k / top-p and ban tokens
def _apply_sampling_controls(
    probs: Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    banned_token_ids: list | None = None,
) -> Tensor:
    """
    probs: [..., V] non-negative and normalized along -1
    Returns filtered, renormalized probs with safeguards.
    """
    V = probs.size(-1)
    probs = probs.clamp_min(1e-12)

    # ban tokens (e.g., mask/pad)
    if banned_token_ids:
        for tok in banned_token_ids:
            if 0 <= tok < V:
                probs[..., tok] = 0.0

    # temperature
    if temperature != 1.0:
        invT = 1.0 / max(1e-6, float(temperature))
        probs = probs.pow(invT)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    # top-k
    if isinstance(top_k, int) and top_k > 0 and top_k < V:
        topk_vals, topk_idx = probs.topk(top_k, dim=-1)
        keep_mask = torch.zeros_like(probs, dtype=torch.bool)
        keep_mask.scatter_(-1, topk_idx, True)
        probs = torch.where(keep_mask, probs, torch.zeros_like(probs))

    # top-p (nucleus)
    if 0.0 < float(top_p) < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cumsum = sorted_probs.cumsum(dim=-1)
        keep = cumsum <= top_p
        # always keep at least the most probable token
        keep[..., 0] = True
        filtered_sorted = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
        probs = torch.zeros_like(probs).scatter(-1, sorted_idx, filtered_sorted)

    # final renorm with uniform fallback if everything got masked
    sums = probs.sum(dim=-1, keepdim=True)
    zero_mask = sums <= 0
    if zero_mask.any():
        uniform = torch.ones_like(probs)
        if banned_token_ids:
            for tok in banned_token_ids:
                if 0 <= tok < V:
                    uniform[..., tok] = 0.0
        uniform = uniform / uniform.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        probs = torch.where(zero_mask, uniform, probs)
        sums = probs.sum(dim=-1, keepdim=True)

    return probs / sums.clamp_min(1e-12)


@torch.inference_mode()
def inference(
    config,
    model,
    duration_model,
    tokenizer,
    sentence,
    nsf: int = 10,
    text_ref: str = None,
    codes_ref: Tensor = None,
    sequence_length: int = 300,
    device: torch.device = torch.device("cuda"),
    # >>> NEW: sampling controls
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    remove_mask_pad_from_choices: bool = False,
) -> Tensor:
    if text_ref is not None:
        augmented_sentence = text_ref + " " + sentence
    else:
        augmented_sentence = sentence
    # text_ids = tokenizer.encode(augmented_sentence, lang="en-us")
    text_ids = tokenizer.encode(augmented_sentence, lang="pt-br")

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
    t_final = 2.0
    time_grid = torch.linspace(t_init, t_final, num_steps + 1, device=device)

    sequence_length = get_remaining_duration(
        duration_model,
        text_ids=text_ids,
        codes_ref=codes_ref,
        device=device
    )

    sequence_length += 50

    # sequence_length = 2048

    print("-"*100)
    print("Predicted sequence length:", sequence_length)
    print("codes_ref.size(0):", codes_ref.size(0))

    # Initialize x_t; for example, using the masked source
    xt = source_distribution.sample((1, sequence_length + codes_ref.size(0)), device=device)

    orig_ref_code_len = codes_ref.size(0)

    num_steps = nsf
    dt = 1.0 / num_steps
    x1_temp = 1.0

    xt[..., : orig_ref_code_len] = codes_ref[..., : orig_ref_code_len]
    text_att_mask = text_ids.new_zeros((1, text_ids.size(1)), dtype=torch.bool)

    # >>> NEW: build banned token list once
    banned_token_ids = []
    if remove_mask_pad_from_choices:
        banned_token_ids.extend([
            int(config.datasets.audio_pad_token),
        ])

    # Loop over the time grid
    for step in range(num_steps):
        t_val = step * dt
        t_tensor = xt.new_full((1,), t_val)
        # Predictor update: compute model output and update x_t
        logits_inf = model(
            x_t=xt,
            text_ids=text_ids,
            text_att_mask=text_att_mask,
            time=t_tensor,
            drop_text=False
        )
        p1 = torch.softmax(logits_inf, dim=-1)
        one_hot_x_t = torch.nn.functional.one_hot(xt, num_classes=vocab_size).float()

        # Compute the velocity update using the denoiser formulation
        u = (p1 - one_hot_x_t) / (1.0 - t_val + 1e-8)

        # Euler update: compute new probabilities
        new_probs = one_hot_x_t + dt * u
        new_probs = new_probs / new_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        # >>> NEW: controlled sampling
        new_probs = _apply_sampling_controls(
            new_probs,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            banned_token_ids=banned_token_ids,
        )
        xt = Categorical(probs=new_probs).sample()

        # Optional: Corrector iterations at the current time step
        for _ in range(num_corrector_steps):
            h_corr = dt * 0.1
            logits_corr = model(
                x_t=xt,
                text_ids=text_ids,
                text_att_mask=text_att_mask,
                time=t_tensor,
                drop_text=False
            )
            p1_corr = torch.softmax(logits_corr, dim=-1)
            one_hot_x_t_corr = torch.nn.functional.one_hot(xt, num_classes=vocab_size).float()

            # Compute the corrector velocity similarly
            u_corr = (p1_corr - one_hot_x_t_corr) / (1.0 - t_val + 1e-8)
            new_probs_corr = one_hot_x_t_corr + h_corr * u_corr
            new_probs_corr = new_probs_corr / new_probs_corr.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            # >>> NEW: controlled sampling (corrector step)
            new_probs_corr = _apply_sampling_controls(
                new_probs_corr,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                banned_token_ids=banned_token_ids,
            )
            xt = Categorical(probs=new_probs_corr).sample()

        xt[..., : orig_ref_code_len] = codes_ref[..., : orig_ref_code_len]

    return xt


@torch.no_grad()
def main() -> None:
    output_dir = "outputs_pred_dur_eos_simple_v3"
    gpu = 0
    config_path = "/raid/aluno_alef/DFM-TTS-2/config/offline-bpe-text_cfg-en.yaml"
    pretrained_checkpoint = "/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/m1ejk3am/checkpoints/epoch=29-step=500000-val/loss_epoch=3.366.ckpt"

    duration_pred_config_path = "/raid/aluno_alef/DFM-TTS-2/config/duration_predictor_bpe_en.yaml"
    duration_pred_pretrained_checkpoint = "/raid/aluno_alef/DFM-TTS-2/src/Duration-Predictor-DFM-TTS/61g87haf/checkpoints/epoch=10-step=138116-val/loss_epoch=4.717.ckpt"

    config = OmegaConf.load(config_path)
    duration_pred_config = OmegaConf.load(duration_pred_config_path)

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    tokenizer = VoiceBpeTokenizer(vocab_file=config.datasets.vocab_file)
    model = DFMTTSWrapper.load_from_checkpoint(pretrained_checkpoint, config=config, map_location=device, strict=False)
    model.eval()

    duration_model = DurationPredictorWrapper.load_from_checkpoint(
        duration_pred_pretrained_checkpoint,
        config=duration_pred_config,
        map_location=device,
        strict=False
    )
    duration_model.eval()

    audio_codec = XCodec2Model.from_pretrained(config.datasets.audio_codec).to(device)
    audio_codec.eval()

    ref_path = "/raid/time_voz/DATASETS_TTS/LibriTTS_R/dev-clean/1462/170138/1462_170138_000001_000004.wav"
    # ref_path = "/raid/aluno_alef/DATASETS/dataset_alc_48k_md5/bbd699/100/a600e123eb.wav"

    audio_ref, audio_ref_sr = torchaudio.load(ref_path)
    if audio_ref_sr != 16000:
        audio_ref = torchaudio.transforms.Resample(audio_ref_sr, 16000)(audio_ref)

    codes_ref = audio_codec.encode_code(input_waveform=audio_ref).squeeze()

    sentences = [
        "Active artists always appreciate artistic achievements and applaud awesome artworks.",
        "Brave bakers boldly baked big batches of brownies in beautiful bakeries.",
        "Daring dancers dazzled during dynamic dance displays, drawing delighted crowds.",
        "Excited engineers eagerly enjoyed exploring enormous engineering exhibits.",
        "Friendly farmers faithfully fostered fields, favoring fruitful crops.",
        "Gallant gophers gracefully gambled golden gooseberries on grandiose glaciers.",
        "Happy hikers harmoniously hiked through hilly landscapes on heavenly holidays."
    ]

    # nsf (steps) list
    nsf = [128, 256, 512, 1024, 2048]

    # >>> NEW: sampling knobs (set as you like)
    temperature = 0.8
    top_k = 0          # e.g., 50
    top_p = 1.0        # e.g., 0.95
    remove_mask_pad_from_choices = True  # switch between original and filtered sampling

    os.makedirs(output_dir, exist_ok=True)

    for idx, sentence in enumerate(sentences):
        print(f"Processing sentence {idx + 1}/{len(sentences)}")
        for n in tqdm(nsf):
            text_ref = "He spoke with an extreme Oxford accent, and when he was talking well, his face sometimes wore the rapt expression of a very emotional man listening to music."
            # text_ref = None
            x_t = inference(
                config=config,
                model=model,
                duration_model=duration_model,
                tokenizer=tokenizer,
                sentence=sentence,
                nsf=n,
                text_ref=text_ref,
                codes_ref=codes_ref,
                device=device,
                # >>> NEW: pass sampling knobs
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                remove_mask_pad_from_choices=remove_mask_pad_from_choices,
            )
            # remove making tokens from the generated sequence
            x_t = x_t.squeeze(0)
            x_t = x_t[codes_ref.size(0):]
            print("1", x_t.shape)
            print("/"*100)
            # count how many eos tokens there are in x_t
            eos_count = (x_t == config.datasets.audio_eos_token).sum().item()
            eos_index = (x_t == config.datasets.audio_eos_token).nonzero(as_tuple=True)[0]
            print(x_t)
            if eos_index.numel() > 0:
                x_t = x_t[..., :eos_index]
            print(f"Number of EOS tokens: {eos_count}, first EOS token at index: {eos_index}")
            x_t = x_t[x_t != config.datasets.audio_eos_token]
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
            torchaudio.save(f"{output_dir}/audio_{idx}-{n}.wav", generated_audio.squeeze(0).cpu(), 16000)


if __name__ == "__main__":
    main()
