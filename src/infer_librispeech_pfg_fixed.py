import os
import json
import logging
import argparse
import warnings
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
    print("Calculating remaining duration...")
    bos_vec = codes_ref.new_full((1,), 65536, dtype=torch.long)
    codes_ref = torch.cat((bos_vec, codes_ref), dim=0)   # [C, dur+1]

    remaining_duration = duration_model(
        text_ids=text_ids,
        audio_ids=codes_ref.unsqueeze(0).to(device)
    )
    return torch.argmax(remaining_duration[:, -1], dim=-1).item()


def cubic_kappa(t: torch.Tensor, a: float = 0.0, b: float = 2.0) -> torch.Tensor:
    return (-2*t**3 + 3*t**2 + a*(t**3 - 2*t**2 + t) + b*(t**3 - t**2)).clamp(0.0, 1.0)

def cubic_kappa_dot(t: torch.Tensor, a: float = 0.0, b: float = 2.0) -> torch.Tensor:
    return (-6*t**2 + 6*t + a*(3*t**2 - 4*t + 1) + b*(3*t**2 - 2*t))

def linear_kappa(t: torch.Tensor) -> torch.Tensor:
    return t

def linear_kappa_dot(t: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(t)

def kappa_and_dot(t: torch.Tensor, kind: str = "cubic", a: float = 0.0, b: float = 2.0) -> tuple[torch.Tensor, torch.Tensor]:
    if kind == "cubic":
        return cubic_kappa(t, a, b), cubic_kappa_dot(t, a, b)
    elif kind == "linear":
        return linear_kappa(t), linear_kappa_dot(t)
    else:
        raise ValueError(f"Unknown kappa kind: {kind}")

# optional: corrector scheduler α_t = 1 + α * t^{a_c} (1-t)^{b_c}
def corrector_alpha_beta(
    tau: torch.Tensor,
    alpha_strength: float = 0.0,
    a_c: float = 0.25,
    b_c: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    at = 1.0 + float(alpha_strength) * (tau**float(a_c)) * ((1.0 - tau)**float(b_c))
    bt = at - 1.0
    return at, bt


@torch.inference_mode()
def inference_pfg(
    config,
    model,
    duration_model,
    tokenizer,
    sentence,
    nsf: int = 10,
    text_ref: str = None,
    codes_ref: torch.Tensor = None,
    sequence_length: Union[int, None] = None,
    device: torch.device = torch.device("cuda"),
    x1_temp: float = 1.0,
    noise: float = 0.0,                 # stochastic remask strength (paper’s “stochasticity”)
    guidance_scale: float = 1.0,        # interpreted as inverse_guide_temp in [0,1] (1=fully conditional)
    alpha_strength: float = 0.0,
    kappa_kind: str = "cubic",
) -> torch.Tensor:
    """
    CTMC-Euler sampler with Predictor-Free Guidance (rates blend) aligned with
    'Unlocking Guidance for Discrete State-Space Diffusion and Flow Models'.
    - Uses κ(t) reparameterization: τ = κ(t), scales rates by κ̇/(1-κ).
    - Blends unconditional/conditional *rates* in log-space with inverse_guide_temp∈[0,1].
    - Optionally forbids transitions into PAD and remasking PADs when PAD/EOS-as-PAD is used.
    - Keeps the reference prefix pinned at every step (robust under noise>0).
    """
    eps = 1e-9

    # --- text ids ---
    augmented_sentence = (text_ref + " " + sentence) if text_ref is not None else sentence
    text_ids = torch.tensor(tokenizer.encode(augmented_sentence, lang="en-us")).unsqueeze(0).to(device)
    text_att_mask = text_ids.new_ones((1, text_ids.size(1)), dtype=torch.bool)

    # --- vocab / tokens ---
    vocab_size = config.datasets.audio_vocab_size + config.model.add_token
    S = vocab_size
    mask_token_id = int(config.datasets.audio_mask_token)
    eos_token_id = int(getattr(config.datasets, "audio_eos_token", -1))
    pad_token_id = None
    if hasattr(config.datasets, "audio_pad_token"):
        pad_token_id = int(config.datasets.audio_pad_token)
    elif getattr(config.datasets, "use_eos_as_pad", False):
        pad_token_id = eos_token_id

    # one-hots we’ll reuse
    mask_one_hot = torch.zeros((S), device=device); mask_one_hot[mask_token_id] = 1.0
    pad_one_hot = None
    if pad_token_id is not None and 0 <= pad_token_id < S:
        pad_one_hot = torch.zeros((S), device=device); pad_one_hot[pad_token_id] = 1.0

    # --- κ scheduler hyperparams (if provided) ---
    kappa_a = float(getattr(getattr(config, "sampler", {}), "kappa_a", 0.0))
    kappa_b = float(getattr(getattr(config, "sampler", {}), "kappa_b", 2.0))
    alpha_a = float(getattr(getattr(config, "sampler", {}), "corrector_a", 0.25))
    alpha_b = float(getattr(getattr(config, "sampler", {}), "corrector_b", 0.5))

    # --- length (duration model or oracle) ---
    if sequence_length is None:
        sequence_length = get_remaining_duration(duration_model, text_ids=text_ids, codes_ref=codes_ref, device=device)

    # --- init x_t with MASKs and pin reference prefix ---
    source_mask = torch.full((1, sequence_length + codes_ref.size(0)), mask_token_id, device=device, dtype=torch.long)
    xt = source_mask
    orig_ref_code_len = int(codes_ref.size(0))
    if codes_ref.size(0) < xt.size(1):
        codes_ref = F.pad(codes_ref, (0, xt.size(1) - codes_ref.size(0)), value=mask_token_id).unsqueeze(0).to(device)
    else:
        codes_ref = codes_ref.unsqueeze(0).to(device)
    xt[..., :orig_ref_code_len] = codes_ref[..., :orig_ref_code_len]

    # --- time grid (uniform in linear t; we use κ̇ for change-of-variable) ---
    num_steps = int(nsf)
    dt_lin = 1.0 / num_steps

    # map guidance_scale -> inverse_guide_temp in [0,1] (paper’s parameter)
    inverse_guide_temp = float(guidance_scale)
    inverse_guide_temp = max(0.0, min(1.0, inverse_guide_temp))  # clamp

    for step in range(num_steps):
        # τ = κ(t), κ̇ = dκ/dt
        t_lin = torch.tensor(step * dt_lin, device=device, dtype=torch.float32)
        tau, kdot = kappa_and_dot(t_lin, kind=kappa_kind, a=kappa_a, b=kappa_b)  # scalars (tensors)
        denom = (1.0 - float(tau)).__float__()  # avoid tiny broadcasting surprises
        denom = max(denom, 1e-6)
        kdot_f = float(kdot.clamp_min(1e-6))

        t_tensor = tau.unsqueeze(0)  # model expects shape [1]

        # ----- unconditional pass -----
        logits_u = model(
            x_t=xt,
            text_ids=text_ids,
            text_att_mask=text_att_mask,
            time=t_tensor,
            drop_text=True
        )
        probs_u = torch.softmax(logits_u / x1_temp, dim=-1)  # [B=1, L, S]

        # ----- conditional pass -----
        logits_c = model(
            x_t=xt,
            text_ids=text_ids,
            text_att_mask=text_att_mask,
            time=t_tensor,
            drop_text=False
        )
        probs_c = torch.softmax(logits_c / x1_temp, dim=-1)  # [1, L, S]

        # ----- rates: unmask when current token is MASK; optional remask elsewhere -----
        xt_mask = (xt == mask_token_id).unsqueeze(-1).float()        # [1, L, 1]
        base_r = (1.0 + float(noise) * float(tau)) * (kdot_f / denom)  # scalar

        # unmasking rates
        R_u = xt_mask * probs_u * base_r
        R_c = xt_mask * probs_c * base_r

        # forbid unmasking into PAD (if configured)
        if pad_one_hot is not None:
            gate = (1.0 - pad_one_hot.view(1, 1, S))
            R_u = R_u * gate
            R_c = R_c * gate

        # remask rates for non-masked coords (scaled by noise)
        if noise > 0.0:
            remask = (1.0 - xt_mask) * mask_one_hot.view(1, 1, S) * float(noise)
            # do NOT remask PAD positions
            if pad_token_id is not None:
                xt_is_pad = (xt == pad_token_id).unsqueeze(-1).float()
                remask = remask * (1.0 - xt_is_pad)
            R_u = R_u + remask
            R_c = R_c + remask

        # ----- PFG: blend rates in log-space with inverse_guide_temp ∈ [0,1] -----
        log_Ru = torch.log(R_u + eps)
        log_Rc = torch.log(R_c + eps)
        R_mix = torch.exp(inverse_guide_temp * log_Rc + (1.0 - inverse_guide_temp) * log_Ru)

        # keep support only on rows that are mask (no off-diagonal where not masked)
        R_mix = R_mix * ((xt == mask_token_id).unsqueeze(-1).float())

        # set diagonal to negative row sum (rate matrix property)
        R_mix.scatter_(-1, xt[..., None], 0.0)
        R_mix.scatter_(-1, xt[..., None], -R_mix.sum(-1, keepdim=True))

        # ----- Euler CTMC step: P ≈ I + Δt R -----
        P = (R_mix * dt_lin).clamp_min(0.0)
        row_off = P.sum(-1, keepdim=True)
        diag = (1.0 - row_off).clamp_min(0.0)
        P.scatter_(-1, xt[..., None], diag)
        P = torch.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
        P = P / P.sum(-1, keepdim=True).clamp_min(1e-12)

        # sample next tokens
        xt = torch.multinomial(P.view(-1, S), 1).view_as(xt)

        # re-pin reference prefix (critical if noise > 0)
        xt[..., :orig_ref_code_len] = codes_ref[..., :orig_ref_code_len]

        # ----- optional small corrector step (nudges towards p1) -----
        if alpha_strength > 0.0:
            alpha_t, _beta_t = corrector_alpha_beta(tau, alpha_strength, alpha_a, alpha_b)
            h_corr = dt_lin * 0.1 * float(alpha_t)

            logits_corr = model(
                x_t=xt,
                text_ids=text_ids,
                text_att_mask=text_att_mask,
                time=t_tensor,
                drop_text=False
            )
            p1_corr = torch.softmax(logits_corr / x1_temp, dim=-1)

            one_hot_x_t_corr = torch.nn.functional.one_hot(xt, num_classes=S).float()
            u_corr = (p1_corr - one_hot_x_t_corr) * (kdot_f / denom)   # (κ̇/(1-κ))*(p1-δx)
            new_probs_corr = one_hot_x_t_corr + h_corr * u_corr
            new_probs_corr = new_probs_corr.clamp_min(0.0)
            new_probs_corr = new_probs_corr / new_probs_corr.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            xt = torch.distributions.Categorical(probs=new_probs_corr).sample()

            # keep prefix fixed after corrector as well
            xt[..., :orig_ref_code_len] = codes_ref[..., :orig_ref_code_len]

    return xt


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
    sequence_length: Union[int, None] = None,
    device: torch.device = torch.device("cuda"),
    x1_temp: float = 1.0,
    alpha_strength: float = 0.0,
) -> Tensor:
    # misc numerics
    eps = 1e-12
    # --- text ids ---
    augmented_sentence = (text_ref + ". " + sentence) if text_ref is not None else sentence
    text_ids = torch.tensor(tokenizer.encode(augmented_sentence, lang="en-us")).unsqueeze(0).to(device)

    max_length = config.datasets.max_audio_length
    vocab_size = config.datasets.audio_vocab_size + config.model.add_token

    # --- source dist (all MASK) ---
    source_distribution = MaskedSourceDistribution(mask_token=config.datasets.audio_mask_token)

    # --- scheduler hyperparams (with safe defaults) ---
    # Path (κ) scheduler
    kappa_a = float(getattr(getattr(config, "sampler", {}), "kappa_a", 0.0))
    kappa_b = float(getattr(getattr(config, "sampler", {}), "kappa_b", 2.0))
    # Corrector α schedule (optional)
    alpha_a = float(getattr(getattr(config, "sampler", {}), "corrector_a", 0.25))
    alpha_b = float(getattr(getattr(config, "sampler", {}), "corrector_b", 0.25))

    # print(f"\n\n\tsequence_length {sequence_length}\n")
    if sequence_length is None:
        # duration for target length
        sequence_length = get_remaining_duration(duration_model, text_ids=text_ids, codes_ref=codes_ref, device=device)

    # init xt with mask then pin the reference prefix
    xt = source_distribution.sample((1, sequence_length + codes_ref.size(0)), device=device)
    orig_ref_code_len = codes_ref.size(0)
    if codes_ref.size(0) < sequence_length + codes_ref.size(0):
        codes_ref = F.pad(codes_ref, (0, sequence_length), value=config.datasets.audio_mask_token).unsqueeze(0)
    xt[..., :orig_ref_code_len] = codes_ref[..., :orig_ref_code_len]

    # constants
    S = vocab_size
    mask_token_id = config.datasets.audio_mask_token
    mask_one_hot = torch.zeros((S), device=model.device); mask_one_hot[mask_token_id] = 1.0
    text_att_mask = text_ids.new_ones((1, text_ids.size(1)), dtype=torch.bool)

    # time grid
    num_steps = int(nsf)
    dt_lin = 1.0 / num_steps

    for step in range(num_steps):
        # linear time -> cubic path time τ = κ(t)
        t_lin = torch.tensor(step * dt_lin, device=device, dtype=torch.float32)
        tau = cubic_kappa(t_lin, a=kappa_a, b=kappa_b)                  # scalar in [0,1]
        kdot = cubic_kappa_dot(t_lin, a=kappa_a, b=kappa_b).clamp_min(1e-6)  # ensure ≥ 0
        t_tensor = tau.unsqueeze(0)  # shape [1] for the model

        # conditional pass
        logits = model(
            x_t=xt,
            text_ids=text_ids,
            text_att_mask=text_att_mask,
            time=t_tensor,
            drop_text=False
        )
        probs = torch.softmax(logits / x1_temp, dim=-1)

        one_hot_x_t = torch.nn.functional.one_hot(xt, num_classes=vocab_size).float()
        u = (probs - one_hot_x_t) / (1.0 - tau)  # forward-time velocity ~ (p1 - δx)/(1-κ)
        h = min(dt_lin, 1.0 - tau)  # step size in τ (ensure we don't step beyond τ=1)
        xt = torch.distributions.Categorical(probs=one_hot_x_t + h * u).sample()

        # --- optional small corrector step (uses κ and α_t) ---
        if alpha_strength > 0.0:
            # α_t schedule at current τ (we only use α_t to scale h_corr here)
            alpha_t, _beta_t = corrector_alpha_beta(tau, alpha_strength, alpha_a, alpha_b)
            h_corr = dt_lin * 0.1 * float(alpha_t)  # scale your existing 10% rule by α_t

            logits_corr = model(
                x_t=xt,
                text_ids=text_ids,
                text_att_mask=text_att_mask,
                time=t_tensor,
                drop_text=False
            )
            p1_corr = torch.softmax(logits_corr / x1_temp, dim=-1)

            one_hot_x_t_corr = torch.nn.functional.one_hot(xt, num_classes=vocab_size).float()
            denom = (1.0 - tau).clamp_min(1e-6)
            u_corr = (p1_corr - one_hot_x_t_corr) / denom  # forward-time velocity ~ (p1 - δx)/(1-κ)
            u_corr = u_corr * float(kdot)                  # scale by κ̇

            new_probs_corr = one_hot_x_t_corr + h_corr * u_corr
            new_probs_corr = new_probs_corr.clamp_min(0.0)
            new_probs_corr = new_probs_corr / new_probs_corr.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            xt = torch.distributions.Categorical(probs=new_probs_corr).sample()
        xt[..., :orig_ref_code_len] = codes_ref[..., :orig_ref_code_len]

    return xt


@torch.no_grad()
def main():
    # inference params
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--alpha_strength", type=float, default=0.0)
    parser.add_argument("--kappa_kind", type=str, choices=["cubic", "linear"], default="cubic", help="Scheduler path κ(t): cubic or linear.")
    args = parser.parse_args()

    base_dir = "/raid/aluno_alef/DFM-TTS-2/src"
    use_oracle_length = True
    nsf = [16, 32, 64, 128, 256, 512, 1024]
    # nsf = [1024]
    noise=args.noise
    guidance_scale=args.guidance_scale
    alpha_strength=args.alpha_strength
    kappa_kind=args.kappa_kind
    libri_speech_test_clean_metadata = "/raid/aluno_alef/DATASETS/LibriSpeech-test-clean-filtered.csv"

    pfg_list = ["m1ejk3am", "xcrhi3ra", "px8ocppp", "fkpl1tsp", "w1kigq88", "b9gp3yjn"]

    ################################################################################
    # select model to evaluate
    ################################################################################

    if args.wandb_id == "2br7dfgc":
        print("\n\n\t Evaluating 2br7dfgc: BPE-EN-eos_as_pad-cubic model \n\n")
        output_dir = f"librispeech-test-clean-filtered/2br7dfgc-bpe-en-eos_as_pad-cubic-use_oracle_length_{use_oracle_length}-noise_{noise}-guidance_scale_{guidance_scale}-alpha_strength_{alpha_strength}-kappa_kind_{kappa_kind}"
        config_path = "/raid/aluno_alef/DFM-TTS-2/config/offline-bpe-text-eos_as_pad-en.yaml"
        pretrained_checkpoint = "/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/2br7dfgc/checkpoints/epoch=23-step=400000-val/loss_epoch=3.386.ckpt"

    elif args.wandb_id == "vf9q9ysg":
        print("\n\n\t Evaluating vf9q9ysg: BPE-EN-eos_as_pad-pad_as_loss-cubic model \n\n")
        output_dir = f"librispeech-test-clean-filtered/vf9q9ysg-bpe-en-eos_as_pad-pad_as_loss-cubic-use_oracle_length_{use_oracle_length}-noise_{noise}-guidance_scale_{guidance_scale}-alpha_strength_{alpha_strength}-kappa_kind_{kappa_kind}"
        config_path = "/raid/aluno_alef/DFM-TTS-2/config/offline-bpe-text-eos_as_pad-pad_loss-en.yaml"
        pretrained_checkpoint = "/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/vf9q9ysg/checkpoints/epoch=23-step=400000-val/loss_epoch=1.214.ckpt"

    elif args.wandb_id == "m1ejk3am":
        print("\n\n\t Evaluating m1ejk3am: BPE-PFG-en-cubic model \n\n")
        output_dir = f"INV-librispeech-test-clean-filtered/m1ejk3am-bpe-pfg-en-cubic-use_oracle_length_{use_oracle_length}-noise_{noise}-guidance_scale_{guidance_scale}-alpha_strength_{alpha_strength}-kappa_kind_{kappa_kind}"
        config_path = "/raid/aluno_alef/DFM-TTS-2/config/offline-bpe-text_cfg-en.yaml"
        pretrained_checkpoint = "/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/m1ejk3am/checkpoints/epoch=29-step=500000-val/loss_epoch=3.366.ckpt"

    elif args.wandb_id == "xcrhi3ra":
        print("\n\n\t Evaluating xcrhi3ra: BPE-PFG-en-eos_as_pad-pad_as_loss-cubic model \n\n")
        output_dir = f"librispeech-test-clean-filtered/xcrhi3ra-bpe-pfg-en-eos_as_pad-pad_as_loss-cubic-use_oracle_length_{use_oracle_length}-noise_{noise}-guidance_scale_{guidance_scale}-alpha_strength_{alpha_strength}-kappa_kind_{kappa_kind}"
        config_path = "/raid/aluno_alef/DFM-TTS-2/config/offline-bpe-text_cfg-eos_as_pad-en.yaml"
        pretrained_checkpoint = "/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/xcrhi3ra/checkpoints/epoch=23-step=400000-val/loss_epoch=1.191.ckpt"

    elif args.wandb_id == "px8ocppp":
        print("\n\n\t Evaluating px8ocppp: BPE-PFG-en-eos_as_pad-weighted-cubic model \n\n")
        output_dir = f"librispeech-test-clean-filtered/px8ocppp-bpe-pfg-en-eos_as_pad_weighted-cubic-use_oracle_length_{use_oracle_length}-noise_{noise}-guidance_scale_{guidance_scale}-alpha_strength_{alpha_strength}-kappa_kind_{kappa_kind}"
        config_path = "/raid/aluno_alef/DFM-TTS-2/config/offline-bpe-text_cfg-eos_as_pad_weighted-en.yaml"
        pretrained_checkpoint = "/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/px8ocppp/checkpoints/epoch=29-step=500000-val/loss_epoch=2.701.ckpt"

    elif args.wandb_id == "fkpl1tsp":
        print("\n\n\t Evaluating fkpl1tsp: FKPL1TSP multilingual BPE-PFG-en-eos_as_pad-pad_as_loss-cubic model \n\n")
        output_dir = f"librispeech-test-clean-filtered/fkpl1tsp-multilingual-bpe-pfg-en-eos_as_pad-pad_as_loss-cubic-corrector-use_oracle_length_{use_oracle_length}-noise_{noise}-guidance_scale_{guidance_scale}-alpha_strength_{alpha_strength}-kappa_kind_{kappa_kind}"
        config_path = "/raid/aluno_alef/DFM-TTS-2/config/offline-bpe-text_cfg-eos_as_pad-multilingual.yaml"
        pretrained_checkpoint = "/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/fkpl1tsp/checkpoints/epoch=11-step=400000-val/loss_epoch=1.737.ckpt"

    elif args.wandb_id == "b9gp3yjn":
        print("\n\n\t Evaluating b9gp3yjn: BPE-PFG-en-eos_as_pad-pad_as_loss-linear model \n\n")
        output_dir = f"librispeech-test-clean-filtered/b9gp3yjn-bpe-pfg-en-eos_as_pad-pad_as_loss-linear-use_oracle_length_{use_oracle_length}-noise_{noise}-guidance_scale_{guidance_scale}-alpha_strength_{alpha_strength}-kappa_kind_{kappa_kind}"
        config_path = "/raid/aluno_alef/DFM-TTS-2/config/offline-bpe-text_cfg-eos_as_pad-en.yaml"
        pretrained_checkpoint = "/raid/aluno_alef/slurm/DFM-TTS/b9gp3yjn/checkpoints/epoch=11-step=200000-val/loss_epoch=2.475.ckpt"

    elif args.wandb_id == "5e51cehx":
        print("\n\n\t Evaluating 5e51cehx: BPE-PFG-en-eos_as_pad-pad_as_loss-linear model \n\n")
        output_dir = f"librispeech-test-clean-filtered/5e51cehx-bpe-pfg-en-eos_as_pad-pad_as_loss-linear-use_oracle_length_{use_oracle_length}-noise_{noise}-guidance_scale_{guidance_scale}-alpha_strength_{alpha_strength}-kappa_kind_{kappa_kind}"
        config_path = "/raid/aluno_alef/DFM-TTS-2/config/offline-bpe-text_cfg-eos_as_pad-en.yaml"
        pretrained_checkpoint = "/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/5e51cehx/checkpoints/epoch=17-step=300000-val/loss_epoch=2.458.ckpt"

    elif args.wandb_id == "u9bejzcm":
        print("\n\n\t Evaluating u9bejzcm: BPE-PFG-en-eos_as_pad-pad_as_loss-linear model \n\n")
        output_dir = f"librispeech-test-clean-filtered/u9bejzcm-bpe-pfg-en-eos_as_pad-pad_as_loss-linear-use_oracle_length_{use_oracle_length}-noise_{noise}-guidance_scale_{guidance_scale}-alpha_strength_{alpha_strength}-kappa_kind_{kappa_kind}"
        config_path = "/raid/aluno_alef/DFM-TTS-2/config/offline-bpe-text_cfg-eos_as_pad-en.yaml"
        pretrained_checkpoint = "/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/u9bejzcm/checkpoints/epoch=11-step=200000-val/loss_epoch=5.700.ckpt"

    elif args.wandb_id == "w1kigq88":
        print("\n\n\t Evaluating w1kigq88: w1kigq88 multilingual BPE-PFG-en-eos_as_pad-pad_as_loss-cubic model \n\n")
        output_dir = f"librispeech-test-clean-filtered/w1kigq88-multilingual-bpe-pfg-en-eos_as_pad-pad_as_loss-cubic-corrector-use_oracle_length_{use_oracle_length}-noise_{noise}-guidance_scale_{guidance_scale}-alpha_strength_{alpha_strength}-kappa_kind_{kappa_kind}"
        config_path = "/raid/aluno_alef/DFM-TTS-2/config/offline-bpe-text_cfg-eos_as_pad-multilingual.yaml"
        pretrained_checkpoint = "/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/w1kigq88/checkpoints/epoch=08-step=300000-val/loss_epoch=1.615.ckpt"

    elif args.wandb_id == "mnporf1f":
        print("\n\n\t Evaluating mnporf1f: mnporf1f multilingual BPE-PFG-en-eos_as_pad-pad_as_loss-cubic model \n\n")
        output_dir = f"v2-librispeech-test-clean-filtered/mnporf1f-multilingual-bpe-pfg-en-eos_as_pad-pad_as_loss-cubic-corrector-use_oracle_length_{use_oracle_length}-noise_{noise}-guidance_scale_{guidance_scale}-alpha_strength_{alpha_strength}-kappa_kind_{kappa_kind}"
        config_path = "/raid/aluno_alef/DFM-TTS-2/config/offline-bpe-text_cfg-eos_as_pad-en.yaml"
        pretrained_checkpoint = "/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/mnporf1f/checkpoints/epoch=11-step=200000-val/loss_epoch=2.752.ckpt"
    else:
        raise ValueError("Invalid wandb_id. Please provide a valid wandb_id.")

    df = pd.read_csv(libri_speech_test_clean_metadata)

    print(df.columns)
    gpu = 0

    config = OmegaConf.load(config_path)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    duration_model = None
    if not use_oracle_length:
        duration_pred_config_path = "/raid/aluno_alef/DFM-TTS-2/config/duration_predictor_bpe_en.yaml"
        duration_pred_pretrained_checkpoint = "/raid/aluno_alef/DFM-TTS-2/src/Duration-Predictor-DFM-TTS/61g87haf/checkpoints/epoch=10-step=138116-val/loss_epoch=4.717.ckpt"

        duration_pred_config = OmegaConf.load(duration_pred_config_path)
        duration_model = DurationPredictorWrapper.load_from_checkpoint(
            duration_pred_pretrained_checkpoint,
            config=duration_pred_config,
            map_location=device,
            strict=False
        )
        duration_model.eval()

    tokenizer = VoiceBpeTokenizer(vocab_file=config.datasets.vocab_file)
    model = DFMTTSWrapper.load_from_checkpoint(pretrained_checkpoint, config=config, map_location=device, strict=False)
    model.eval()

    audio_codec = XCodec2Model.from_pretrained(config.datasets.audio_codec).to(device)
    audio_codec.eval()

    os.makedirs(output_dir, exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            text = row["text"]
            text_ref = row["ref_text"]

            filepath_codec = row["filepath_codec"]
            ref_filepath_codec = row["reference_codec"]

            codes_ref = torch.load(ref_filepath_codec).squeeze().to(device)

            oracle_length = None
            if use_oracle_length:
                oracle_codes = torch.load(filepath_codec).squeeze()
                oracle_length = oracle_codes.shape[-1]

            for n in tqdm(nsf):
                output_filepath = os.path.join(base_dir, output_dir, f"audio_{idx}-{n}.wav")
                if os.path.exists(output_filepath):
                    # print(f"File {output_filepath} already exists, skipping...")
                    continue

                if args.wandb_id in pfg_list:
                    x_t = inference_pfg(
                        config=config,
                        model=model,
                        duration_model=duration_model,
                        tokenizer=tokenizer,
                        sentence=text,
                        nsf=n,
                        text_ref=text_ref,
                        codes_ref=codes_ref,
                        device=device,
                        sequence_length=oracle_length if use_oracle_length else None,
                        noise=noise,
                        guidance_scale=guidance_scale,
                        alpha_strength=alpha_strength,
                        kappa_kind=kappa_kind,
                    )
                else:
                    x_t = inference(
                        config=config,
                        model=model,
                        duration_model=duration_model,
                        tokenizer=tokenizer,
                        sentence=text,
                        nsf=n,
                        text_ref=text_ref,
                        codes_ref=codes_ref,
                        device=device,
                        sequence_length=oracle_length if use_oracle_length else None,
                    )
                # remove making tokens from the generated sequence
                x_t = x_t.squeeze(0)
                x_t = x_t[codes_ref.size(0):]

                x_t = x_t[x_t != config.datasets.audio_eos_token]
                x_t = x_t[x_t != config.datasets.audio_mask_token]
                # remove padding tokens from the generated sequence
                if hasattr(config.datasets, "audio_pad_token"):
                    x_t = x_t[x_t != config.datasets.audio_pad_token]
                x_t = x_t.unsqueeze(0).unsqueeze(0)
                # Decode the final token sequence into an audio waveform
                generated_audio = audio_codec.decode_code(x_t)
                torchaudio.save(output_filepath, generated_audio.squeeze(0).cpu(), 16000)
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue


if __name__ == "__main__":
    main()
