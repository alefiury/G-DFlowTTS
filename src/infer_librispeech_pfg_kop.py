# KOP-compliant Predictor-Free Guidance inference for discrete flow matching
# (Flow Matching with General Discrete Paths: A Kinetic-Optimal Perspective)

import os
import json
import logging
import argparse
import warnings
from pprint import pprint
from typing import Tuple, Union, Optional

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


# ------------------------------------------------------------------------------------
# Source distribution (Dirac at MASK)
# ------------------------------------------------------------------------------------
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


# ------------------------------------------------------------------------------------
# Duration helper (unchanged)
# ------------------------------------------------------------------------------------
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


# ------------------------------------------------------------------------------------
# Schedulers: cubic/linear (existing) + KOP geodesic (NEW)
# ------------------------------------------------------------------------------------
def cubic_kappa(t: torch.Tensor, a: float = 0.0, b: float = 2.0) -> torch.Tensor:
    return (-2*t**3 + 3*t**2 + a*(t**3 - 2*t**2 + t) + b*(t**3 - t**2)).clamp(0.0, 1.0)

def cubic_kappa_dot(t: torch.Tensor, a: float = 0.0, b: float = 2.0) -> torch.Tensor:
    return (-6*t**2 + 6*t + a*(3*t**2 - 4*t + 1) + b*(3*t**2 - 2*t))

def linear_kappa(t: torch.Tensor) -> torch.Tensor:
    return t

def linear_kappa_dot(t: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(t)

# --- KOP masked-source geodesic: κ(t) = sin^2(π t / 2), κ̇(t) = (π/2) sin(π t)
def kop_kappa(t: torch.Tensor) -> torch.Tensor:
    return torch.sin(0.5 * torch.pi * t) ** 2

def kop_kappa_dot(t: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.pi * torch.sin(torch.pi * t)

def kappa_and_dot(t: torch.Tensor, kind: str = "cubic", a: float = 0.0, b: float = 2.0) -> tuple[torch.Tensor, torch.Tensor]:
    if kind == "cubic":
        return cubic_kappa(t, a, b), cubic_kappa_dot(t, a, b)
    elif kind == "linear":
        return linear_kappa(t), linear_kappa_dot(t)
    elif kind == "kop":
        return kop_kappa(t), kop_kappa_dot(t)
    else:
        raise ValueError(f"Unknown kappa kind: {kind}")


# ------------------------------------------------------------------------------------
# Optional corrector schedule (unchanged)
# ------------------------------------------------------------------------------------
def corrector_alpha_beta(
    tau: torch.Tensor,
    alpha_strength: float = 0.0,
    a_c: float = 0.25,
    b_c: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    at = 1.0 + float(alpha_strength) * (tau**float(a_c)) * ((1.0 - tau)**float(b_c))
    bt = at - 1.0
    return at, bt


# ------------------------------------------------------------------------------------
# Small helpers for schedules, + NEW KOP-safe time step
# ------------------------------------------------------------------------------------
import math

def _interp_w(tau: float, kind: str = "cosine", power: float = 1.0) -> float:
    if kind == "cosine":
        return 0.5 * (1.0 - math.cos(math.pi * float(tau)))
    if kind == "linear":
        return float(tau)
    if kind == "poly":
        return float(tau) ** float(power)
    if kind == "sigmoid":
        k = 10.0
        s = 1.0 / (1.0 + math.exp(-k * (float(tau) - 0.5)))
        s0 = 1.0 / (1.0 + math.exp(+k * 0.5))  # τ=0
        s1 = 1.0 - s0                           # τ=1
        return (s - s0) / (s1 - s0 + 1e-12)
    raise ValueError(f"Unknown schedule kind: {kind}")

def _sched(tau: float, start: float, end: float, kind: str = "cosine", power: float = 1.0) -> float:
    return float(start) + (float(end) - float(start)) * _interp_w(tau, kind=kind, power=power)

# --- NEW: KOP step-size guard to keep (I + dt R) row-stochastic & avoid κ overshoot
def _dt_eff_from_rates(R: torch.Tensor, dt_lin: float, kappa_f: float, kdot_f: float) -> float:
    """
    Choose a safe dt:
      (i) don't overshoot κ→1: dt ≤ (1-κ)/κ̇
     (ii) keep (I + dt R) row-stochastic: dt ≤ 0.999 / max_row_out_rate
    """
    dt_kappa = (1.0 - float(kappa_f)) / max(float(kdot_f), 1e-12)
    out_rate = R.clamp_min(0).sum(-1)              # [B,L]
    rmax = float(out_rate.max().item()) if out_rate.numel() else 0.0
    dt_rate = 0.999 / max(rmax, 1e-12) if rmax > 0 else dt_lin
    return float(min(dt_lin, dt_kappa, dt_rate))


# ------------------------------------------------------------------------------------
# Predictor-Free Guidance (KOP-compliant)
# ------------------------------------------------------------------------------------
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
    sequence_length: Optional[int] = None,
    device: torch.device = torch.device("cuda"),
    # constants (no scheduling on these)
    x1_temp: float = 1.0,                 # base temperature τ; scheduled via τ_t = τ * (1 - t)^2
    noise: float = 0.0,                   # constant stochastic remask strength
    guidance_scale: float = 1.0,          # mixed in log-rate space; keep your semantics
    alpha_strength: float = 0.0,
    kappa_kind: str = "kop",              # "kop" | "cubic" | "linear"
    integrator: str = "heun",             # "euler" | "midpoint" | "heun"
) -> torch.Tensor:
    """
    CTMC τ-leaping with Predictor-Free Guidance and higher-order integrators, KOP-compliant.
    - Path time κ(t) via kappa_kind; hazard λ = κ̇/(1-κ) scales jump rates.
    - Adaptive dt keeps PMFs valid and prevents κ overshoot.
    - Prefix is re-pinned every substep; PAD gating is enforced.
    """
    eps = 1e-9

    # ----- temperature schedule (Eq. 36) over *linear* time t ∈ [0,1] -----
    def _temp_at_linear_t(t_lin: float) -> float:
        return max(1e-3, float(x1_temp) * (1.0 - float(t_lin))**2)

    # ---------- text ----------
    augmented_sentence = (text_ref + " " + sentence) if text_ref is not None else sentence
    text_ids = torch.tensor(tokenizer.encode(augmented_sentence, lang="en-us")).unsqueeze(0).to(device)
    text_att_mask = text_ids.new_ones((1, text_ids.size(1)), dtype=torch.bool)

    # ---------- tokens / sizes ----------
    S = int(config.datasets.audio_vocab_size + config.model.add_token)
    mask_token_id = int(config.datasets.audio_mask_token)
    eos_token_id  = int(getattr(config.datasets, "audio_eos_token", -1))
    pad_token_id  = None
    if hasattr(config.datasets, "audio_pad_token"):
        pad_token_id = int(config.datasets.audio_pad_token)
    elif getattr(config.datasets, "use_eos_as_pad", False):
        pad_token_id = eos_token_id

    mask_one_hot = torch.zeros((S), device=device); mask_one_hot[mask_token_id] = 1.0
    pad_one_hot = None
    if pad_token_id is not None and 0 <= pad_token_id < S:
        pad_one_hot = torch.zeros((S), device=device); pad_one_hot[pad_token_id] = 1.0

    # ---------- κ scheduler & corrector hyperparams ----------
    kappa_a = float(getattr(getattr(config, "sampler", {}), "kappa_a", 0.0))
    kappa_b = float(getattr(getattr(config, "sampler", {}), "kappa_b", 2.0))
    alpha_a = float(getattr(getattr(config, "sampler", {}), "corrector_a", 0.25))
    alpha_b = float(getattr(getattr(config, "sampler", {}), "corrector_b", 0.5))

    # length
    if sequence_length is None:
        sequence_length = get_remaining_duration(duration_model, text_ids=text_ids, codes_ref=codes_ref, device=device)

    # init xt + pin prefix
    xt = torch.full((1, sequence_length + codes_ref.size(0)), mask_token_id, device=device, dtype=torch.long)
    orig_ref_code_len = int(codes_ref.size(0))
    if codes_ref.size(0) < xt.size(1):
        codes_ref = F.pad(codes_ref, (0, xt.size(1) - codes_ref.size(0)), value=mask_token_id).unsqueeze(0).to(device)
    else:
        codes_ref = codes_ref.unsqueeze(0).to(device)
    xt[..., :orig_ref_code_len] = codes_ref[..., :orig_ref_code_len]

    # time grid
    num_steps = int(nsf)
    dt_lin = 1.0 / num_steps

    # PFG mix strength (same semantics as your code: mix in log-rate space)
    inv_guide = max(0.0, min(5.0, float(guidance_scale)))
    noise_c   = float(noise)

    # ---- core builders ----
    def _rates_blend(xt_cur: torch.Tensor, kappa_f: float, kdot_f: float, T_x1: float) -> torch.Tensor:
        # model "time" input: use κ(t) (keeps consistency with earlier inference code)
        t_tensor = torch.tensor([kappa_f], device=device, dtype=torch.float32)

        # unconditional
        logits_u = model(x_t=xt_cur, text_ids=text_ids, text_att_mask=text_att_mask, time=t_tensor, drop_text=True)
        probs_u  = torch.softmax(logits_u / T_x1, dim=-1)

        # conditional
        logits_c = model(x_t=xt_cur, text_ids=text_ids, text_att_mask=text_att_mask, time=t_tensor, drop_text=False)
        probs_c  = torch.softmax(logits_c / T_x1, dim=-1)

        xt_mask  = (xt_cur == mask_token_id).unsqueeze(-1).float()

        # KOP hazard scale λ = κ̇/(1-κ)
        denom    = max(1.0 - kappa_f, 1e-6)
        base_r   = (1.0 + noise_c * kappa_f) * (kdot_f / denom)

        # propose unmasking rates
        R_u = xt_mask * probs_u * base_r
        R_c = xt_mask * probs_c * base_r

        # forbid unmasking into PAD
        if pad_one_hot is not None:
            gate = (1.0 - pad_one_hot.view(1, 1, S))
            R_u = R_u * gate
            R_c = R_c * gate

        # optional stochastic remasking elsewhere (if noise>0)
        if noise_c > 0.0:
            remask = (1.0 - xt_mask) * mask_one_hot.view(1, 1, S) * noise_c
            if pad_token_id is not None:
                xt_is_pad = (xt_cur == pad_token_id).unsqueeze(-1).float()
                remask = remask * (1.0 - xt_is_pad)
            R_u = R_u + remask
            R_c = R_c + remask

        # PFG mix in log-rate space
        log_Ru = torch.log(R_u + eps)
        log_Rc = torch.log(R_c + eps)
        R_mix  = torch.exp(inv_guide * log_Rc + (1.0 - inv_guide) * log_Ru)

        # keep support only when masked
        R_mix = R_mix * ((xt_cur == mask_token_id).unsqueeze(-1).float())

        # zero diag then set diag = -row_sum
        R_mix.scatter_(-1, xt_cur[..., None], 0.0)
        R_mix.scatter_(-1, xt_cur[..., None], -R_mix.sum(-1, keepdim=True))
        return R_mix

    def _step_from_rates(xt_cur: torch.Tensor, R: torch.Tensor, dt_eff: float) -> torch.Tensor:
        P = (R * dt_eff).clamp_min(0.0)
        row_off = P.sum(-1, keepdim=True)
        diag = (1.0 - row_off).clamp_min(0.0)
        P.scatter_(-1, xt_cur[..., None], diag)
        P = torch.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
        P = P / P.sum(-1, keepdim=True).clamp_min(1e-12)
        return torch.multinomial(P.view(-1, S), 1).view_as(xt_cur)

    # choose integrator
    integrator = integrator.lower()
    assert integrator in {"euler", "midpoint", "heun"}, "integrator must be 'euler', 'midpoint', or 'heun'"

    for step in range(num_steps):
        t0_lin  = step * dt_lin
        # κ(t0), κ̇(t0)
        kappa0, kdot0 = kappa_and_dot(torch.tensor(t0_lin, device=device), kind=kappa_kind, a=kappa_a, b=kappa_b)
        kappa0_f  = float(kappa0); kdot0_f = float(kdot0.clamp_min(1e-6))
        # Eq. 36 temperature at *linear* time
        T0      = _temp_at_linear_t(t0_lin)

        if integrator == "euler":
            R0      = _rates_blend(xt, kappa0_f, kdot0_f, T0)
            dt_eff  = _dt_eff_from_rates(R0, dt_lin, kappa0_f, kdot0_f)
            xt      = _step_from_rates(xt, R0, dt_eff)
            xt[..., :orig_ref_code_len] = codes_ref[..., :orig_ref_code_len]

        elif integrator == "midpoint":
            # pilot half-step
            R0      = _rates_blend(xt, kappa0_f, kdot0_f, T0)
            dt_half = _dt_eff_from_rates(R0, dt_lin * 0.5, kappa0_f, kdot0_f)
            x_half  = _step_from_rates(xt, R0, dt_half)
            x_half[..., :orig_ref_code_len] = codes_ref[..., :orig_ref_code_len]

            t_mid_lin = t0_lin + 0.5 * dt_lin
            kappa_mid, kdot_mid = kappa_and_dot(torch.tensor(t_mid_lin, device=device), kind=kappa_kind, a=kappa_a, b=kappa_b)
            kappa_mid_f  = float(kappa_mid); kdot_mid_f = float(kdot_mid.clamp_min(1e-6))
            T_mid      = _temp_at_linear_t(t_mid_lin)

            R_mid   = _rates_blend(x_half, kappa_mid_f, kdot_mid_f, T_mid)
            dt_eff  = _dt_eff_from_rates(R_mid, dt_lin, kappa_mid_f, kdot_mid_f)
            xt      = _step_from_rates(xt, R_mid, dt_eff)
            xt[..., :orig_ref_code_len] = codes_ref[..., :orig_ref_code_len]

        else:  # "heun"
            # predictor: Euler full-step
            R0      = _rates_blend(xt, kappa0_f, kdot0_f, T0)
            dt_eul  = _dt_eff_from_rates(R0, dt_lin, kappa0_f, kdot0_f)
            x_tilde = _step_from_rates(xt, R0, dt_eul)
            x_tilde[..., :orig_ref_code_len] = codes_ref[..., :orig_ref_code_len]

            # evaluate at t1
            t1_lin = t0_lin + dt_lin
            kappa1, kdot1 = kappa_and_dot(torch.tensor(t1_lin, device=device), kind=kappa_kind, a=kappa_a, b=kappa_b)
            kappa1_f  = float(kappa1); kdot1_f = float(kdot1.clamp_min(1e-6))
            T1      = _temp_at_linear_t(t1_lin)
            R1      = _rates_blend(x_tilde, kappa1_f, kdot1_f, T1)

            # trapezoid average
            R_bar = 0.5 * (R0 + R1)
            R_bar.scatter_(-1, xt[..., None], 0.0)
            R_bar.scatter_(-1, xt[..., None], -R_bar.sum(-1, keepdim=True))

            dt_eff = _dt_eff_from_rates(R_bar, dt_lin, kappa0_f, kdot0_f)
            xt     = _step_from_rates(xt, R_bar, dt_eff)
            xt[..., :orig_ref_code_len] = codes_ref[..., :orig_ref_code_len]

        # optional corrector (one-term)
        if alpha_strength > 0.0:
            alpha_t, _ = corrector_alpha_beta(torch.tensor(t0_lin, device=device), alpha_strength, alpha_a, alpha_b)
            h_corr = dt_lin * 0.1 * float(alpha_t)

            denom = max(1.0 - float(kappa0), 1e-6)
            t_tensor = torch.tensor([float(kappa0)], device=device, dtype=torch.float32)
            logits = model(x_t=xt, text_ids=text_ids, text_att_mask=text_att_mask, time=t_tensor, drop_text=False)
            p1 = torch.softmax(logits / _temp_at_linear_t(t0_lin), dim=-1)

            one_hot = F.one_hot(xt, num_classes=S).float()
            u_corr = (p1 - one_hot) * (float(kdot0.clamp_min(1e-6)) / denom)  # = λ(t)*(p1-δx)
            probs = (one_hot + h_corr * u_corr).clamp_min(0.0)
            probs = probs / probs.sum(-1, keepdim=True).clamp_min(1e-12)
            xt = torch.distributions.Categorical(probs=probs).sample()
            xt[..., :orig_ref_code_len] = codes_ref[..., :orig_ref_code_len]

    return xt


# ------------------------------------------------------------------------------------
# Simple Euler inference (no PFG) — made KOP-consistent
# ------------------------------------------------------------------------------------
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
    kappa_kind: str = "kop",            # NEW: allow kop here too
) -> Tensor:
    eps = 1e-12

    # text ids
    augmented_sentence = (text_ref + ". " + sentence) if text_ref is not None else sentence
    text_ids = torch.tensor(tokenizer.encode(augmented_sentence, lang="en-us")).unsqueeze(0).to(device)
    text_att_mask = text_ids.new_ones((1, text_ids.size(1)), dtype=torch.bool)

    max_length = config.datasets.max_audio_length
    vocab_size = config.datasets.audio_vocab_size + config.model.add_token

    # source dist
    source_distribution = MaskedSourceDistribution(mask_token=config.datasets.audio_mask_token)

    # scheduler hyperparams
    kappa_a = float(getattr(getattr(config, "sampler", {}), "kappa_a", 0.0))
    kappa_b = float(getattr(getattr(config, "sampler", {}), "kappa_b", 2.0))
    alpha_a = float(getattr(getattr(config, "sampler", {}), "corrector_a", 0.25))
    alpha_b = float(getattr(getattr(config, "sampler", {}), "corrector_b", 0.25))

    if sequence_length is None:
        sequence_length = get_remaining_duration(duration_model, text_ids=text_ids, codes_ref=codes_ref, device=device)

    # init xt with mask then pin the reference prefix
    xt = source_distribution.sample((1, sequence_length + codes_ref.size(0)), device=device)
    orig_ref_code_len = codes_ref.size(0)
    if codes_ref.size(0) < sequence_length + codes_ref.size(0):
        codes_ref = F.pad(codes_ref, (0, sequence_length), value=config.datasets.audio_mask_token).unsqueeze(0)
    xt[..., :orig_ref_code_len] = codes_ref[..., :orig_ref_code_len]

    # constants
    S = vocab_size

    # time grid
    num_steps = int(nsf)
    dt_lin = 1.0 / num_steps

    for step in range(num_steps):
        # linear time -> path time κ(t)
        t_lin = torch.tensor(step * dt_lin, device=device, dtype=torch.float32)
        kappa, kdot = kappa_and_dot(t_lin, kind=kappa_kind, a=kappa_a, b=kappa_b)
        kappa = kappa.clamp(0.0, 1.0)
        kdot  = kdot.clamp_min(1e-6)
        t_tensor = kappa.unsqueeze(0)  # model sees κ(t)

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

        # KOP velocity: v = λ(t)*(p1 - δx), with λ = κ̇/(1-κ)
        lam = (kdot / (1.0 - kappa).clamp_min(1e-6)).item()
        Rv  = lam * (probs - one_hot_x_t)

        # KOP step bounds
        dt_kappa = float((1.0 - kappa).item()) / float(kdot.item())
        out_rate = Rv.clamp_min(0).sum(-1)             # [B,L]
        rmax = float(out_rate.max().item()) if out_rate.numel() else 0.0
        dt_rate = 0.999 / max(rmax, 1e-12) if rmax > 0 else float(dt_lin)
        h = min(float(dt_lin), dt_kappa, dt_rate)

        new_probs = (one_hot_x_t + h * Rv).clamp_min(0.0)
        new_probs = new_probs / new_probs.sum(-1, keepdim=True).clamp_min(1e-12)
        xt = torch.distributions.Categorical(probs=new_probs).sample()

        # --- optional small corrector step (uses κ and α_t) ---
        if alpha_strength > 0.0:
            # α_t schedule at current *linear* time
            alpha_t, _beta_t = corrector_alpha_beta(t_lin, alpha_strength, alpha_a, alpha_b)
            h_corr = dt_lin * 0.1 * float(alpha_t)

            logits_corr = model(
                x_t=xt,
                text_ids=text_ids,
                text_att_mask=text_att_mask,
                time=t_tensor,
                drop_text=False
            )
            p1_corr = torch.softmax(logits_corr / x1_temp, dim=-1)

            one_hot_x_t_corr = torch.nn.functional.one_hot(xt, num_classes=vocab_size).float()
            denom = (1.0 - kappa).clamp_min(1e-6)
            u_corr = (p1_corr - one_hot_x_t_corr) * (kdot / denom)  # λ(t)*(p1-δx)

            new_probs_corr = (one_hot_x_t_corr + h_corr * u_corr).clamp_min(0.0)
            new_probs_corr = new_probs_corr / new_probs_corr.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            xt = torch.distributions.Categorical(probs=new_probs_corr).sample()

        xt[..., :orig_ref_code_len] = codes_ref[..., :orig_ref_code_len]

    return xt


# ------------------------------------------------------------------------------------
# CLI / main (minimal edits to expose KOP)
# ------------------------------------------------------------------------------------
@torch.no_grad()
def main():
    # inference params
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--alpha_strength", type=float, default=0.0)
    parser.add_argument("--kappa_kind", type=str, choices=["kop", "cubic", "linear"], default="kop",
                        help="Path κ(t): 'kop' (sin^2), 'cubic', or 'linear'.")
    parser.add_argument("--integrator", type=str, choices=["euler", "midpoint", "heun"], default="euler",
                        help="CTMC integrator: Euler, Midpoint, or Heun.")
    args = parser.parse_args()

    base_dir = "/raid/aluno_alef/DFM-TTS-2/src"
    use_oracle_length = True
    nsf = [16, 32, 64, 128, 256, 512, 1024]
    noise=args.noise
    guidance_scale=args.guidance_scale
    alpha_strength=args.alpha_strength
    kappa_kind=args.kappa_kind
    libri_speech_test_clean_metadata = "/raid/aluno_alef/DATASETS/LibriSpeech-test-clean-filtered.csv"
    integrator = args.integrator

    pfg_list = ["m1ejk3am", "xcrhi3ra", "px8ocppp", "fkpl1tsp", "w1kigq88", "b9gp3yjn", "mnporf1f", "mv7mrkk9"]

    ################################################################################
    # select model to evaluate (unchanged paths)
    ################################################################################
    if args.wandb_id == "2br7dfgc":
        print("\n\n\t Evaluating 2br7dfgc: BPE-EN-eos_as_pad-cubic model \n\n")
        output_dir = f"librispeech-test-clean-filtered/2br7dfgc-bpe-en-eos_as_pad-{kappa_kind}-use_oracle_length_{use_oracle_length}-noise_{noise}-guidance_scale_{guidance_scale}-alpha_strength_{alpha_strength}-kappa_kind_{kappa_kind}"
        config_path = "/raid/aluno_alef/DFM-TTS-2/config/offline-bpe-text-eos_as_pad-en.yaml"
        pretrained_checkpoint = "/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/2br7dfgc/checkpoints/epoch=23-step=400000-val/loss_epoch=3.386.ckpt"

    elif args.wandb_id == "vf9q9ysg":
        print("\n\n\t Evaluating vf9q9ysg: BPE-EN-eos_as_pad-pad_as_loss-cubic model \n\n")
        output_dir = f"librispeech-test-clean-filtered/vf9q9ysg-bpe-en-eos_as_pad-pad_as_loss-{kappa_kind}-use_oracle_length_{use_oracle_length}-noise_{noise}-guidance_scale_{guidance_scale}-alpha_strength_{alpha_strength}-kappa_kind_{kappa_kind}"
        config_path = "/raid/aluno_alef/DFM-TTS-2/config/offline-bpe-text-eos_as_pad-pad_loss-en.yaml"
        pretrained_checkpoint = "/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/vf9q9ysg/checkpoints/epoch=23-step=400000-val/loss_epoch=1.214.ckpt"

    elif args.wandb_id == "m1ejk3am":
        print("\n\n\t Evaluating m1ejk3am: BPE-PFG-en model \n\n")
        output_dir = f"INV-librispeech-test-clean-filtered/m1ejk3am-bpe-pfg-en-{kappa_kind}-use_oracle_length_{use_oracle_length}-noise_{noise}-guidance_scale_{guidance_scale}-alpha_strength_{alpha_strength}-kappa_kind_{kappa_kind}"
        config_path = "/raid/aluno_alef/DFM-TTS-2/config/offline-bpe-text_cfg-en.yaml"
        pretrained_checkpoint = "/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/m1ejk3am/checkpoints/epoch=29-step=500000-val/loss_epoch=3.366.ckpt"

    elif args.wandb_id == "xcrhi3ra":
        print("\n\n\t Evaluating xcrhi3ra: BPE-PFG-en-eos_as_pad-pad_as_loss model \n\n")
        output_dir = f"librispeech-test-clean-filtered/xcrhi3ra-bpe-pfg-en-eos_as_pad-pad_as_loss-{kappa_kind}-use_oracle_length_{use_oracle_length}-noise_{noise}-guidance_scale_{guidance_scale}-alpha_strength_{alpha_strength}-kappa_kind_{kappa_kind}"
        config_path = "/raid/aluno_alef/DFM-TTS-2/config/offline-bpe-text_cfg-eos_as_pad-en.yaml"
        pretrained_checkpoint = "/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/xcrhi3ra/checkpoints/epoch=23-step=400000-val/loss_epoch=1.191.ckpt"

    elif args.wandb_id == "px8ocppp":
        print("\n\n\t Evaluating px8ocppp: BPE-PFG-en-eos_as_pad-weighted model \n\n")
        output_dir = f"librispeech-test-clean-filtered/px8ocppp-bpe-pfg-en-eos_as_pad_weighted-{kappa_kind}-use_oracle_length_{use_oracle_length}-noise_{noise}-guidance_scale_{guidance_scale}-alpha_strength_{alpha_strength}-kappa_kind_{kappa_kind}"
        config_path = "/raid/aluno_alef/DFM-TTS-2/config/offline-bpe-text_cfg-eos_as_pad_weighted-en.yaml"
        pretrained_checkpoint = "/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/px8ocppp/checkpoints/epoch=29-step=500000-val/loss_epoch=2.701.ckpt"

    elif args.wandb_id == "fkpl1tsp":
        print("\n\n\t Evaluating fkpl1tsp: multilingual BPE-PFG-en-eos_as_pad-pad_as_loss model \n\n")
        output_dir = f"librispeech-test-clean-filtered/fkpl1tsp-multilingual-bpe-pfg-en-eos_as_pad-pad_as_loss-{kappa_kind}-corrector-use_oracle_length_{use_oracle_length}-noise_{noise}-guidance_scale_{guidance_scale}-alpha_strength_{alpha_strength}"
        config_path = "/raid/aluno_alef/DFM-TTS-2/config/offline-bpe-text_cfg-eos_as_pad-multilingual.yaml"
        pretrained_checkpoint = "/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/fkpl1tsp/checkpoints/epoch=11-step=400000-val/loss_epoch=1.737.ckpt"

    elif args.wandb_id == "b9gp3yjn":
        print("\n\n\t Evaluating b9gp3yjn: BPE-PFG-en-eos_as_pad-pad_as_loss-linear model \n\n")
        output_dir = f"librispeech-test-clean-filtered/b9gp3yjn-bpe-pfg-en-eos_as_pad-pad_as_loss-{kappa_kind}-use_oracle_length_{use_oracle_length}-noise_{noise}-guidance_scale_{guidance_scale}-alpha_strength_{alpha_strength}"
        config_path = "/raid/aluno_alef/DFM-TTS-2/config/offline-bpe-text_cfg-eos_as_pad-en.yaml"
        pretrained_checkpoint = "/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/b9gp3yjn/checkpoints/epoch=11-step=200000-val/loss_epoch=2.475.ckpt"

    elif args.wandb_id == "5e51cehx":
        print("\n\n\t Evaluating 5e51cehx: BPE-PFG-en-eos_as_pad-pad_as_loss-linear model \n\n")
        output_dir = f"librispeech-test-clean-filtered/5e51cehx-bpe-pfg-en-eos_as_pad-pad_as_loss-{kappa_kind}-use_oracle_length_{use_oracle_length}-noise_{noise}-guidance_scale_{guidance_scale}-alpha_strength_{alpha_strength}"
        config_path = "/raid/aluno_alef/DFM-TTS-2/config/offline-bpe-text_cfg-eos_as_pad-en.yaml"
        pretrained_checkpoint = "/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/5e51cehx/checkpoints/epoch=17-step=300000-val/loss_epoch=2.458.ckpt"

    elif args.wandb_id == "u9bejzcm":
        print("\n\n\t Evaluating u9bejzcm: BPE-PFG-en-eos_as_pad-pad_as_loss-linear model \n\n")
        output_dir = f"librispeech-test-clean-filtered/u9bejzcm-bpe-pfg-en-eos_as_pad-pad_as_loss-{kappa_kind}-use_oracle_length_{use_oracle_length}-noise_{noise}-guidance_scale_{guidance_scale}-alpha_strength_{alpha_strength}"
        config_path = "/raid/aluno_alef/DFM-TTS-2/config/offline-bpe-text_cfg-eos_as_pad-en.yaml"
        pretrained_checkpoint = "/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/u9bejzcm/checkpoints/epoch=11-step=200000-val/loss_epoch=5.700.ckpt"

    elif args.wandb_id == "w1kigq88":
        print("\n\n\t Evaluating w1kigq88: multilingual BPE-PFG-en-eos_as_pad-pad_as_loss-cubic model \n\n")
        output_dir = f"librispeech-test-clean-filtered/w1kigq88-multilingual-bpe-pfg-en-eos_as_pad-pad_as_loss-{kappa_kind}-corrector-use_oracle_length_{use_oracle_length}-noise_{noise}-guidance_scale_{guidance_scale}-alpha_strength_{alpha_strength}"
        config_path = "/raid/aluno_alef/DFM-TTS-2/config/offline-bpe-text_cfg-eos_as_pad-multilingual.yaml"
        pretrained_checkpoint = "/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/w1kigq88/checkpoints/epoch=08-step=300000-val/loss_epoch=1.615.ckpt"

    elif args.wandb_id == "mnporf1f":
        print("\n\n\t Evaluating mnporf1f: multilingual BPE-PFG-en-eos_as_pad-pad_as_loss-cubic model \n\n")
        output_dir = f"v3-librispeech-test-clean-filtered/mnporf1f-multilingual-bpe-pfg-en-eos_as_pad-pad_as_loss-{kappa_kind}-corrector-use_oracle_length_{use_oracle_length}-noise_{noise}-guidance_scale_{guidance_scale}-integrator_{integrator}"
        config_path = "/raid/aluno_alef/DFM-TTS-2/config/offline-bpe-text_cfg-eos_as_pad-en.yaml"
        pretrained_checkpoint = "/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/mnporf1f/checkpoints/epoch=11-step=200000-val/loss_epoch=2.752.ckpt"

    elif args.wandb_id == "mv7mrkk9":
        print("\n\n\t Evaluating mv7mrkk9: multilingual BPE-PFG-en-eos_as_pad-pad_as_loss-cubic model \n\n")
        output_dir = f"v3-librispeech-test-clean-filtered/mv7mrkk9-multilingual-bpe-pfg-en-eos_as_pad-pad_as_loss-{kappa_kind}-corrector-use_oracle_length_{use_oracle_length}-noise_{noise}-guidance_scale_{guidance_scale}-integrator_{integrator}"
        config_path = "/raid/aluno_alef/DFM-TTS-2/config/offline-bpe-text_cfg-eos_as_pad-en-kinect.yaml"
        pretrained_checkpoint = "/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/mv7mrkk9/checkpoints/epoch=11-step=200000-val/loss_epoch=9.656.ckpt"
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
                        integrator=args.integrator,
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
                        kappa_kind=kappa_kind,
                    )
                # strip prefix and special tokens
                x_t = x_t.squeeze(0)
                x_t = x_t[codes_ref.size(1) if codes_ref.dim() == 2 else codes_ref.size(0):]
                x_t = x_t[x_t != config.datasets.audio_eos_token]
                x_t = x_t[x_t != config.datasets.audio_mask_token]
                if hasattr(config.datasets, "audio_pad_token"):
                    x_t = x_t[x_t != config.datasets.audio_pad_token]
                x_t = x_t.unsqueeze(0).unsqueeze(0)

                # Decode to waveform and save
                generated_audio = audio_codec.decode_code(x_t)
                torchaudio.save(output_filepath, generated_audio.squeeze(0).cpu(), 16000)
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue


if __name__ == "__main__":
    main()
