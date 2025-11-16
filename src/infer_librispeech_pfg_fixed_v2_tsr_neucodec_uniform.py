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
from dataclasses import dataclass

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

@dataclass
class SchedulerOutput:
    alpha_t: Tensor
    sigma_t: Tensor
    d_alpha_t: Tensor
    d_sigma_t: Tensor


class KOConvexScheduler:
    """
    KO scheduler used during training, reimplemented for inference.

    alpha_t, sigma_t are the convex coefficients (cos^2, sin^2).
    d_alpha_t, d_sigma_t are their derivatives wrt t.
    """

    def __call__(self, t: Tensor) -> SchedulerOutput:
        """
        t: Tensor in [0, 1], shape e.g. [B] or [].
        """
        # Cos/sin arguments
        # s = 0.5 * pi * (1 - t)
        s = 0.5 * math.pi * (1.0 - t)

        alpha_t = torch.cos(s) ** 2
        sigma_t = torch.sin(s) ** 2

        # d/dt cos^2(s) = 0.5 * pi * sin(pi * (1 - t))
        d_alpha_t = 0.5 * math.pi * torch.sin(math.pi * (1.0 - t))
        # d/dt sin^2(s) = - d/dt cos^2(s)
        d_sigma_t = -d_alpha_t

        return SchedulerOutput(
            alpha_t=alpha_t,
            sigma_t=sigma_t,
            d_alpha_t=d_alpha_t,
            d_sigma_t=d_sigma_t,
        )

    def kappa_inverse(self, kappa: Tensor) -> Tensor:
        """
        Inverse mapping κ -> t, mirroring the training code.
        Not strictly needed for sampling, but kept for completeness.
        """
        return 1.0 - torch.arccos(torch.sqrt(kappa)) * 0.5 / math.pi

# Single global instance is fine for inference
KO_SCHEDULER = KOConvexScheduler()

def kappa_and_dot(t: torch.Tensor, kind: str = "cubic", a: float = 0.0, b: float = 2.0) -> tuple[torch.Tensor, torch.Tensor]:
    if kind == "cubic":
        return cubic_kappa(t, a, b), cubic_kappa_dot(t, a, b)
    elif kind == "linear":
        return linear_kappa(t), linear_kappa_dot(t)
    elif kind == "ko":
        sched = KO_SCHEDULER(t)          # t may be scalar or a vector
        return sched.alpha_t, sched.d_alpha_t
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


import math
import torch
import torch.nn.functional as F

# ---------- tiny helpers ----------
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

@torch.inference_mode()
def inference_pfg(
    config,
    model,
    duration_model,
    tokenizer,
    sentence,
    nsf: int = 10,
    text_ref: str | None = None,
    codes_ref: torch.Tensor | None = None,
    sequence_length: int | None = None,
    device: torch.device = torch.device("cuda"),
    # base constants (no scheduling except temperature)
    x1_temp: float = 1.0,                 # base temperature τ in Eq.36
    noise: float = 0.0,                   # fixed re-noising level
    guidance_scale: float = 1.0,          # PFG mix weight in [0,1]; 1 = fully cond
    alpha_strength: float = 0.0,          # optional one-term corrector (0 = off)
    kappa_kind: str = "cubic",
    # integrator choice
    integrator: str = "euler",            # "euler" | "midpoint" | "heun"
    # temperature scheduling (Eq.36 from discrete flow matching): T(t)=x1_temp*(1-t)^2
    use_dfm36_temp: bool = True,
    # Temporal Score Rescaling (TSR)
    tsr_k: float = 0.93,                  # k>1 sharper / k<1 flatter; k=1 disables TSR
    tsr_sigma: float = 3.0,               # “σ” knob controlling when TSR kicks in
    # NEW: which source distribution we are mimicking at inference time
    prior_type: str | None = None,        # "mask" | "uniform" | None(auto from config)
) -> torch.Tensor:
    """
    CTMC τ-leaping with Predictor-Free Guidance, higher-order integrators, Eq.36
    temperature scheduling, and Temporal Score Rescaling (TSR).

    Supports both:
      • prior_type="mask"   : Mask-absorbing prior (only MASK positions update).
      • prior_type="uniform": Uniform prior over discrete vocab (excluding PAD),
                              like UniformSourceDistribution used in training.
    """
    eps = 1e-9

    # -------------------------------------------------------
    # 0) Decide which prior we are emulating (mask vs uniform)
    # -------------------------------------------------------
    if prior_type is None:
        prior_type = getattr(config, "source_dist_type", "mask")
    prior_type = prior_type.lower()
    assert prior_type in ("mask", "uniform"), \
        f"prior_type must be 'mask' or 'uniform', got: {prior_type}"
    uniform_prior = (prior_type == "uniform")

    # ---------- text ----------
    augmented_sentence = (text_ref + " " + sentence) if text_ref is not None else sentence
    text_ids = torch.tensor(tokenizer.encode(augmented_sentence, lang="en-us")).unsqueeze(0).to(device)
    text_att_mask = text_ids.new_ones((1, text_ids.size(1)), dtype=torch.bool)

    # ---------- tokens / sizes ----------
    S = int(config.datasets.audio_vocab_size + config.model.audio_add_token)

    # Mask / EOS / PAD ids (some may not be used in uniform mode, but are harmless)
    mask_token_id = int(getattr(config.datasets, "audio_mask_token", -1))
    eos_token_id  = int(getattr(config.datasets, "audio_eos_token", -1))

    pad_token_id = None
    if hasattr(config.datasets, "audio_pad_token"):
        pad_token_id = int(config.datasets.audio_pad_token)
    elif getattr(config.datasets, "use_eos_as_pad", False) and eos_token_id >= 0:
        pad_token_id = eos_token_id

    # one-hot helpers
    mask_one_hot = None
    if 0 <= mask_token_id < S:
        mask_one_hot = torch.zeros((S), device=device)
        mask_one_hot[mask_token_id] = 1.0

    pad_one_hot = None
    if pad_token_id is not None and 0 <= pad_token_id < S:
        pad_one_hot = torch.zeros((S), device=device)
        pad_one_hot[pad_token_id] = 1.0

    # ---------- κ scheduler (time change for CTMC) ----------
    kappa_a = float(getattr(getattr(config, "sampler", {}), "kappa_a", 0.0))
    kappa_b = float(getattr(getattr(config, "sampler", {}), "kappa_b", 2.0))
    alpha_a = float(getattr(getattr(config, "sampler", {}), "corrector_a", 0.25))
    alpha_b = float(getattr(getattr(config, "sampler", {}), "corrector_b", 0.5))

    # length
    if codes_ref is None:
        raise ValueError("codes_ref must be provided.")
    orig_ref_code_len = codes_ref.shape[-1]

    print("\n\n==================== Inference PFG Settings ===================")
    print("orig_ref_code_len:", orig_ref_code_len)

    if sequence_length is None:
        sequence_length = get_remaining_duration(
            duration_model,
            text_ids=text_ids,
            codes_ref=codes_ref,
            device=device
        )

    # -------------------------------------------------------
    # 1) Initialize x_0
    #    • MASK prior   : all tokens = mask_token_id
    #    • UNIFORM prior: uniform over [0, S-1), excluding PAD via S-1 like training
    # -------------------------------------------------------
    total_len = sequence_length + orig_ref_code_len
    if uniform_prior:
        # training used UniformSourceDistribution(vocab_size=S-1) to avoid PAD
        xt = torch.randint(
            low=0,
            high=S,
            size=(1, total_len),
            device=device,
            dtype=torch.long,
        )
        xt[..., :orig_ref_code_len] = codes_ref

        # print(xt)
        # print(codes_ref)

    else:
        # original mask-based prior
        if mask_token_id < 0:
            raise ValueError("audio_mask_token must be defined for mask prior.")
        xt = torch.full(
            (1, total_len),
            mask_token_id,
            device=device,
            dtype=torch.long,
        )

        # print("SHAPE CODES REF:", codes_ref.shape)
        # print("SHAPE XT:", xt.shape)
        # # pin the reference prefix
        # xt[..., :orig_ref_code_len] = codes_ref[..., :orig_ref_code_len]

    # time grid
    num_steps = int(nsf)
    dt_lin = 1.0 / max(1, num_steps)

    # ---------- temperature schedule (Eq.36, DFM) ----------
    def _temp_at(t_lin: float) -> float:
        # DFM Eq.36 temperature scheduler: τ_t = τ * (1 - t)^2
        if use_dfm36_temp:
            return float(x1_temp) * max(0.0, (1.0 - t_lin))**2
        return float(x1_temp)

    # ---------- TSR scaling factor (Xu et al. 2025) ----------
    def _tsr_r(t_lin: float) -> float:
        if tsr_k == 1.0:
            return 1.0
        t = max(1e-5, min(1.0 - 1e-5, float(t_lin)))  # avoid div by zero
        snr = ((1.0 - t) ** 2) / (t ** 2)             # η_t for α=1-t, σ=t
        num = snr * (tsr_sigma ** 2) + 1.0
        den = snr * (tsr_sigma ** 2) / max(1e-8, tsr_k) + 1.0
        return num / den

    # -------------------------------------------------------
    # 2) Core: build blended rate matrix R_mix given xt and τ
    #    • mask prior   : only MASK positions emit transitions; remasking possible
    #    • uniform prior: fully dense CTMC over all tokens (off-diagonal only),
    #                     using model probs with PAD column removed.
    # -------------------------------------------------------
    def _rates_blend(
        xt_cur: torch.Tensor,
        t_lin: float,
        tau_f: float,
        kdot_f: float,
        T_x1: float,
        inv_g: float,
        noise_f: float,
    ) -> torch.Tensor:
        """
        Return a CTMC generator R of shape [B, T, S] with row-sum zero.
        """
        t_tensor = torch.tensor([tau_f], device=device, dtype=torch.float32)
        r_t = _tsr_r(t_lin)

        # unconditional / conditional logits (apply TSR by scaling logits)
        logits_u = model(
            x_t=xt_cur,
            text_ids=text_ids,
            text_att_mask=text_att_mask,
            time=t_tensor,
            drop_text=True,
        )
        logits_c = model(
            x_t=xt_cur,
            text_ids=text_ids,
            text_att_mask=text_att_mask,
            time=t_tensor,
            drop_text=False,
        )
        if r_t != 1.0:
            logits_u = logits_u * r_t
            logits_c = logits_c * r_t

        # temperature-softmax to get p1
        probs_u = torch.softmax(logits_u / max(1e-6, T_x1), dim=-1)
        probs_c = torch.softmax(logits_c / max(1e-6, T_x1), dim=-1)

        # base hazard λ(t)
        denom = max(1.0 - float(tau_f), 1e-6)
        base_r = (1.0 + noise_f * float(tau_f)) * (kdot_f / denom)

        inv_g = float(max(0.0, min(1.0, inv_g)))  # clamp in [0,1]

        if uniform_prior:
            # ---- UNIFORM PRIOR BRANCH ---------------------------------------
            # forbid jumps into PAD (like generate_sample_pfg_uniform)
            if pad_one_hot is not None:
                gate = (1.0 - pad_one_hot.view(1, 1, S))
                probs_u = probs_u * gate
                probs_c = probs_c * gate

            # off-diagonal distributions q(i→j) ∝ p1(j), j≠i
            one_hot_cur = F.one_hot(xt_cur, num_classes=S).float()
            off_u = probs_u * (1.0 - one_hot_cur)
            off_u = off_u / off_u.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            off_c = probs_c * (1.0 - one_hot_cur)
            off_c = off_c / off_c.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            # convert to instantaneous rates
            R_u = base_r * off_u
            R_c = base_r * off_c

            # Predictor-Free Guidance in log-rate space
            log_Ru = torch.log(R_u + eps)
            log_Rc = torch.log(R_c + eps)
            R_mix = torch.exp(inv_g * log_Rc + (1.0 - inv_g) * log_Ru)

            # enforce CTMC row-sum zero: diagonal = -sum(off-diag)
            R_off = R_mix.clone()
            R_off.scatter_(-1, xt_cur[..., None], 0.0)
            row_sum = R_off.sum(dim=-1, keepdim=True)
            R = R_off.clone()
            R.scatter_(-1, xt_cur[..., None], -row_sum)
            return R

        else:
            # ---- MASK PRIOR BRANCH (original behaviour) ---------------------
            if mask_one_hot is None:
                raise ValueError("mask_one_hot must be defined for mask prior.")
            xt_mask = (xt_cur == mask_token_id).unsqueeze(-1).float()

            # unmasking rates (only from MASK)
            R_u = xt_mask * probs_u * base_r
            R_c = xt_mask * probs_c * base_r

            # forbid unmasking into PAD
            if pad_one_hot is not None:
                gate = (1.0 - pad_one_hot.view(1, 1, S))
                R_u = R_u * gate
                R_c = R_c * gate

            # remask elsewhere (optional)
            if noise_f > 0.0:
                remask = (1.0 - xt_mask) * mask_one_hot.view(1, 1, S) * noise_f
                if pad_token_id is not None:
                    xt_is_pad = (xt_cur == pad_token_id).unsqueeze(-1).float()
                    remask = remask * (1.0 - xt_is_pad)
                R_u = R_u + remask
                R_c = R_c + remask

            # PFG mix in log-rate space
            log_Ru = torch.log(R_u + eps)
            log_Rc = torch.log(R_c + eps)
            R_mix = torch.exp(inv_g * log_Rc + (1.0 - inv_g) * log_Ru)

            # keep support only when masked
            R_mix = R_mix * xt_mask

            # generator fix: zero diag then set to -row_sum
            R_mix.scatter_(-1, xt_cur[..., None], 0.0)
            R_mix.scatter_(-1, xt_cur[..., None], -R_mix.sum(-1, keepdim=True))
            return R_mix

    # -------------------------------------------------------
    # 3) Single CTMC step given rate matrix R
    # -------------------------------------------------------
    def _step_from_rates(xt_cur: torch.Tensor, R: torch.Tensor, dt: float) -> torch.Tensor:
        P = (R * dt).clamp_min(0.0)
        row_off = P.sum(-1, keepdim=True)
        diag = (1.0 - row_off).clamp_min(0.0)
        P.scatter_(-1, xt_cur[..., None], diag)
        P = torch.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
        P = P / P.sum(-1, keepdim=True).clamp_min(1e-12)
        return torch.multinomial(P.view(-1, S), 1).view_as(xt_cur)

    # -------------------------------------------------------
    # 4) Choose integrator and simulate
    # -------------------------------------------------------
    integrator = integrator.lower()
    assert integrator in {"euler", "midpoint", "heun"}, \
        "integrator must be 'euler', 'midpoint', or 'heun'"

    for step in range(num_steps):
        t0_lin = step * dt_lin
        tau0, kdot0 = kappa_and_dot(
            torch.tensor(t0_lin, device=device),
            kind=kappa_kind,
            a=kappa_a,
            b=kappa_b,
        )
        tau0_f  = float(tau0)
        kdot0_f = float(kdot0.clamp_min(1e-6))
        T0      = _temp_at(t0_lin)
        invg0   = float(guidance_scale)
        noise0  = float(noise)

        if integrator == "euler":
            R0 = _rates_blend(xt, t0_lin, tau0_f, kdot0_f, T0, invg0, noise0)
            xt = _step_from_rates(xt, R0, dt_lin)
            xt[..., :orig_ref_code_len] = codes_ref[..., :orig_ref_code_len]

        elif integrator == "midpoint":
            # pilot half-step
            R0    = _rates_blend(xt, t0_lin, tau0_f, kdot0_f, T0, invg0, noise0)
            xhalf = _step_from_rates(xt, R0, dt_lin * 0.5)
            xhalf[..., :orig_ref_code_len] = codes_ref[..., :orig_ref_code_len]

            t_mid_lin = t0_lin + 0.5 * dt_lin
            taum, kdotm = kappa_and_dot(
                torch.tensor(t_mid_lin, device=device),
                kind=kappa_kind,
                a=kappa_a,
                b=kappa_b,
            )
            Rm = _rates_blend(
                xhalf,
                t_mid_lin,
                float(taum),
                float(kdotm.clamp_min(1e-6)),
                _temp_at(t_mid_lin),
                invg0,
                noise0,
            )
            xt = _step_from_rates(xt, Rm, dt_lin)
            xt[..., :orig_ref_code_len] = codes_ref[..., :orig_ref_code_len]

        elif integrator == "heun":
            R0 = _rates_blend(xt, t0_lin, tau0_f, kdot0_f, T0, invg0, noise0)
            x_tilde = _step_from_rates(xt, R0, dt_lin)
            x_tilde[..., :orig_ref_code_len] = codes_ref[..., :orig_ref_code_len]

            t1_lin = t0_lin + dt_lin
            tau1, kdot1 = kappa_and_dot(
                torch.tensor(t1_lin, device=device),
                kind=kappa_kind,
                a=kappa_a,
                b=kappa_b,
            )
            R1 = _rates_blend(
                x_tilde,
                t1_lin,
                float(tau1),
                float(kdot1.clamp_min(1e-6)),
                _temp_at(t1_lin),
                invg0,
                noise0,
            )
            R_bar = 0.5 * (R0 + R1)
            R_bar.scatter_(-1, xt[..., None], 0.0)
            R_bar.scatter_(-1, xt[..., None], -R_bar.sum(-1, keepdim=True))

            xt = _step_from_rates(xt, R_bar, dt_lin)
            xt[..., :orig_ref_code_len] = codes_ref[..., :orig_ref_code_len]

        # ---------------------------------------------------
        # 5) Optional one-term corrector (same for both priors)
        # ---------------------------------------------------
        if alpha_strength > 0.0:
            alpha_t, _ = corrector_alpha_beta(
                torch.tensor(tau0_f, device=device),
                alpha_strength,
                alpha_a,
                alpha_b,
            )
            h_corr = dt_lin * 0.1 * float(alpha_t)
            denom = max(1.0 - tau0_f, 1e-6)
            t_tensor = torch.tensor([tau0_f], device=device, dtype=torch.float32)

            logits = model(
                x_t=xt,
                text_ids=text_ids,
                text_att_mask=text_att_mask,
                time=t_tensor,
                drop_text=False,
            )
            r_t = _tsr_r(t0_lin)
            if r_t != 1.0:
                logits = logits * r_t
            p1 = torch.softmax(logits / max(1e-6, _temp_at(t0_lin)), dim=-1)

            one_hot = F.one_hot(xt, num_classes=S).float()
            u_corr = (p1 - one_hot) * (float(kdot0.clamp_min(1e-6)) / denom)
            probs = (one_hot + h_corr * u_corr).clamp_min(0.0)
            probs = probs / probs.sum(-1, keepdim=True).clamp_min(1e-12)
            xt = torch.distributions.Categorical(probs=probs).sample()
            xt[..., :orig_ref_code_len] = codes_ref[..., :orig_ref_code_len]

    # make sure prefix is pinned
    xt[..., :orig_ref_code_len] = codes_ref
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
    # inference params
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--noise", type=float, default=0.3)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--alpha_strength", type=float, default=20.0)
    parser.add_argument("--kappa_kind", type=str, choices=["cubic", "linear", "ko"], default="ko", help="Scheduler path κ(t): cubic or linear.")
    parser.add_argument("--integrator", type=str, choices=["euler", "midpoint", "heun"], default="euler", help="CTMC integrator: Euler, Midpoint, or Heun.")
    args = parser.parse_args()

    base_dir = "/raid/aluno_alef/DFM-TTS-2/src"
    use_oracle_length = True
    nsf = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    # nsf = [1024]
    noise=args.noise
    guidance_scale=args.guidance_scale
    alpha_strength=args.alpha_strength
    kappa_kind=args.kappa_kind
    libri_speech_test_clean_metadata = "/raid/aluno_alef/DATASETS/LibriSpeech-test-clean-filtered.csv"
    integrator = args.integrator

    pfg_list = ["m1ejk3am", "xcrhi3ra", "px8ocppp", "fkpl1tsp", "w1kigq88", "b9gp3yjn", "mnporf1f", "nsrtslsi", "igrz0zvk"]

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
        output_dir = f"v4-librispeech-test-clean-filtered/mnporf1f-multilingual-bpe-pfg-en-eos_as_pad-pad_as_loss-cubic-corrector-use_oracle_length_{use_oracle_length}-noise_{noise}-guidance_scale_{guidance_scale}-alpha_strength_{alpha_strength}-kappa_kind_{kappa_kind}-integrator_{integrator}"
        config_path = "/raid/aluno_alef/DFM-TTS-2/config/offline-bpe-text_cfg-eos_as_pad-en.yaml"
        pretrained_checkpoint = "/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/mnporf1f/checkpoints/epoch=11-step=200000-val/loss_epoch=2.752.ckpt"

    elif args.wandb_id == "nsrtslsi":
        print("\n\n\t Evaluating nsrtslsi: nsrtslsi multilingual BPE-PFG-en-eos_as_pad-pad_as_loss-cubic model \n\n")
        output_dir = f"70khours-emilia-yodas-tsr-librispeech-test-clean-filtered/nsrtslsi-multilingual-bpe-pfg-en-eos_as_pad-pad_as_loss-cubic-corrector-use_oracle_length_{use_oracle_length}-noise_{noise}-guidance_scale_{guidance_scale}-alpha_strength_{alpha_strength}-kappa_kind_{kappa_kind}-integrator_{integrator}"
        config_path = "/raid/aluno_alef/DFM-TTS-2/config/offline-bpe-text_cfg-eos_as_pad-en-emilia_yodas.yaml"
        pretrained_checkpoint = "/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/nsrtslsi/checkpoints/epoch=02-step=530000-val/loss_epoch=2.715.ckpt"

    elif args.wandb_id == "igrz0zvk":
        print("\n\n\t Evaluating igrz0zvk: igrz0zvk multilingual BPE-PFG-en-eos_as_pad-pad_as_loss-cubic model \n\n")
        output_dir = f"igrz0zvk-70khours-emilia-yodas-tsr-librispeech-test-clean-filtered/igrz0zvk-multilingual-bpe-pfg-en-eos_as_pad-pad_as_loss-cubic-corrector-use_oracle_length_{use_oracle_length}-noise_{noise}-guidance_scale_{guidance_scale}-alpha_strength_{alpha_strength}-kappa_kind_{kappa_kind}-integrator_{integrator}"
        config_path = "/raid/aluno_alef/DFM-TTS-2/config/offline-bpe-text_cfg-eos_as_pad-en-emilia_yodas_ko.yaml"
        pretrained_checkpoint = "/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/igrz0zvk/checkpoints/epoch=02-step=660000-val/loss_epoch=1.941.ckpt"
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

    # audio_codec = XCodec2Model.from_pretrained(config.datasets.audio_codec).to(device)
    # audio_codec.eval()

    audio_codec = NeuCodec.from_pretrained("neuphonic/neucodec").to(device)
    saving_sr = 24000
    audio_codec.eval()

    os.makedirs(output_dir, exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # try:
        text = row["text"]
        text_ref = row["ref_text"]

        # filepath_codec = row["filepath_codec"]
        # ref_filepath_codec = row["reference_codec"]

        # codes_ref = torch.load(ref_filepath_codec).squeeze().to(device)

        codes_ref = _encode_audio(audio_codec, row["reference"]).to(device)
        print(f"\n\nCodes ref shape: {codes_ref.shape}\n\n")

        # # decode and save the reconstructed reference audio for sanity check
        # reconstructed_ref_audio = audio_codec.decode_code(codes_ref.long().to(device))
        # torchaudio.save(os.path.join(base_dir, output_dir, f"reconstructed_ref_audio_{idx}.wav"), reconstructed_ref_audio.squeeze(0).cpu(), saving_sr)


        oracle_length = None
        if use_oracle_length:
            # oracle_codes = torch.load(filepath_codec).squeeze()
            # oracle_length = oracle_codes.shape[-1]
            oracle_codes = _encode_audio(audio_codec, row["filepath"]).to(device)
            oracle_length = oracle_codes.shape[-1]

            print(f"\n\nOracle length: {oracle_length}\n\n")

        for n in tqdm(nsf):
            output_filepath = os.path.join(base_dir, output_dir, f"audio_{idx}-{n}.wav")
            if os.path.exists(output_filepath):
                # print(f"File {output_filepath} already exists, skipping...")
                continue

            if args.wandb_id in pfg_list:
                print("\n\nUsing PFG inference...\n\n")
                x_t = inference_pfg(
                    config=config,
                    model=model,
                    duration_model=duration_model,
                    tokenizer=tokenizer,
                    sentence=text,
                    nsf=n,
                    text_ref=text_ref,
                    codes_ref=codes_ref.squeeze(0),
                    device=device,
                    sequence_length=oracle_length if use_oracle_length else None,
                    noise=noise,
                    guidance_scale=guidance_scale,
                    alpha_strength=alpha_strength,
                    kappa_kind=kappa_kind,
                    integrator=args.integrator,
                    prior_type="uniform",
                    # use_dfm36_temp=False,
                    # # Temporal Score Rescaling (TSR)
                    # tsr_k=1.0,                  # k>1 sharper / k<1 flatter; k=1 disables TSR
                    # tsr_sigma=1.0,               # “σ” knob controlling when TSR kicks in
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
                    prior_type="uniform",
                )

            mask_token_id   = getattr(config.datasets, "audio_mask_token", None)
            eos_token_id    = getattr(config.datasets, "audio_eos_token", None)
            pad_token_id    = getattr(config.datasets, "audio_pad_token", None)
            expand_token_id = getattr(config.datasets, "audio_expand_token", None)
            delete_token_id = getattr(config.datasets, "audio_delete_token", None)
            # remove making tokens from the generated sequence
            x_t = x_t.squeeze(0)

            print("\n\nGenerated codes before cleaning:", x_t.shape, codes_ref.shape[-1])
            # save ref_audio
            ref_audio = audio_codec.decode_code(x_t[:codes_ref.shape[-1]].unsqueeze(0).unsqueeze(0).long().to(device))
            torchaudio.save(os.path.join(base_dir, output_dir, f"ref_audio_{idx}.wav"), ref_audio.squeeze(0).cpu(), saving_sr)

            x_t = x_t[codes_ref.shape[-1]:]

            print(f"\n\n1 - Generated codes shape: {x_t.shape} | ref codes shape: {codes_ref.shape} | oracle length: {oracle_length}\n\n")

            # x_t = x_t[x_t != config.datasets.audio_eos_token]
            # x_t = x_t[x_t != config.datasets.audio_mask_token]

            if eos_token_id is not None:
                x_t = x_t[x_t != eos_token_id]
                print("Shape after eos removal:", x_t.shape)
            if mask_token_id is not None:
                x_t = x_t[x_t != mask_token_id]
                print("Shape after mask removal:", x_t.shape)
            if pad_token_id is not None:
                x_t = x_t[x_t != pad_token_id]
                print("Shape after pad removal:", x_t.shape)
            if expand_token_id is not None:
                x_t = x_t[x_t != expand_token_id]
                print("Shape after expand removal:", x_t.shape)
            if delete_token_id is not None:
                x_t = x_t[x_t != delete_token_id]
                print("Shape after delete removal:", x_t.shape)

            print(f"\n\n2 - Generated codes shape: {x_t.shape} | ref codes shape: {codes_ref.shape} | oracle length: {oracle_length}\n\n")
            # remove padding tokens from the generated sequence
            if hasattr(config.datasets, "audio_pad_token"):
                x_t = x_t[x_t != config.datasets.audio_pad_token]
            x_t = x_t.unsqueeze(0).unsqueeze(0)
            # Decode the final token sequence into an audio waveform
            generated_audio = audio_codec.decode_code(x_t.long().to(device))
            torchaudio.save(output_filepath, generated_audio.squeeze(0).cpu(), saving_sr)

        # except Exception as e:
        #     print(f"Error processing row {idx}: {e}")
        #     continue


if __name__ == "__main__":
    main()
