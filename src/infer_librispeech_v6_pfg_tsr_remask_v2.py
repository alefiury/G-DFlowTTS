#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DFM-TTS inference (mask-source, MixtureDiscreteProbPath) with:
- Correct CTMC tau-leap sampler (jump prob = 1 - exp(-dt * hazard))
- Optional Predictor-Free Guidance (PFG) mixing in log-rate space
- Optional Temporal Score Rescaling (TSR) applied to BOTH conditional/unconditional logits
- ReMDM-style remasking ("Remasking Discrete Diffusion Models with Inference-Time Scaling")
    * Uses sigma_max constraint from alpha_t, alpha_s
    * Supports "switch" (tswitch) + "rescale" (eta_rescale, eta_cap)
    * Implemented as a CTMC token -> MASK rate so it integrates naturally into hazard/jumps
    * Optional confidence-based remask weighting (low-confidence tokens remask more)

IMPORTANT FIXES (as requested):
- ReMDM remasking NEVER remasks the prefix (prompt) tokens.
- ReMDM remasking NEVER remasks a "fixed" EOS token at the final position when you pin EOS.
- Optional: you can disable pinning the final EOS to *test EOS timing inference*.

Notes:
- If you enable --use_remdm, you usually want --remask_noise 0.0 (since ReMDM provides principled remasking).
- Early stopping is disabled when remasking is enabled (because later remasks can still fix early mistakes).
"""

import os
import time
import math
import argparse
import warnings
from typing import Optional, List, Tuple

warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import pandas as pd
from safetensors.torch import load_file
import torchaudio
from torchaudio import transforms as T
from omegaconf import OmegaConf

from transformers import AutoTokenizer
from tqdm import tqdm

# Your modules
from modules.pl_wrapper import DFMTTSWrapper
from modules.dp_wrapper import DurationPredictorWrapper
from utils.tokenizer import VoiceBpeTokenizer

# Flow-matching path/scheduler
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler

# Optional KO scheduler if you used it
try:
    from modules.flow import KOConvexScheduler
except Exception:
    KOConvexScheduler = None

# Codecs
try:
    from xcodec2.modeling_xcodec2 import XCodec2Model
except Exception:
    XCodec2Model = None

try:
    from neucodec import NeuCodec
except Exception:
    NeuCodec = None


# ----------------------------
# Utilities
# ----------------------------

def seed_everything(seed: int = 0) -> None:
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def ensure_gpt2_padding(tok: AutoTokenizer) -> AutoTokenizer:
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": tok.eos_token})
    return tok


def build_text_inputs(config, sentence: str, text_ref: Optional[str], device: torch.device):
    augmented = (text_ref + ". " + sentence) if (text_ref is not None and len(text_ref) > 0) else sentence

    if config.datasets.type == "hf_text_tokenizer":
        tok = AutoTokenizer.from_pretrained(config.datasets.text_tokenizer_name)
        tok = ensure_gpt2_padding(tok)
        enc = tok(augmented, return_tensors="pt")
        text_ids = enc["input_ids"].to(device)
        text_att_mask = enc["attention_mask"].bool().to(device)
        return text_ids, text_att_mask, tok
    else:
        tok = VoiceBpeTokenizer(vocab_file=config.datasets.vocab_file)
        text_ids = torch.tensor(tok.encode(augmented, lang="en-us"), device=device).unsqueeze(0)
        text_att_mask = torch.ones_like(text_ids, dtype=torch.bool, device=device)
        return text_ids, text_att_mask, tok


def truncate_at_first_eos(tokens_1d: torch.Tensor, eos_id: Optional[int]) -> torch.Tensor:
    if eos_id is None:
        return tokens_1d
    pos = (tokens_1d == eos_id).nonzero(as_tuple=False)
    if pos.numel() == 0:
        return tokens_1d
    first = int(pos[0].item())
    return tokens_1d[:first]


def load_codec(config, device: torch.device):
    codec_name = getattr(config.datasets, "codec_name", "").lower()

    if codec_name == "xcodec2":
        if XCodec2Model is None:
            raise RuntimeError("xcodec2 is not available in this environment.")
        codec = XCodec2Model.from_pretrained(config.datasets.audio_codec).to(device)
        codec.eval()
        saving_sr = int(codec.config.sampling_rate)
        return codec, saving_sr

    if codec_name == "neucodec":
        if NeuCodec is None:
            raise RuntimeError("neucodec is not available in this environment.")
        codec = NeuCodec.from_pretrained("neuphonic/neucodec").to(device)
        codec.eval()
        saving_sr = 24000
        return codec, saving_sr

    raise ValueError(f"Unsupported codec_name: {codec_name}")


def build_path_from_config(config):
    sched_type = str(getattr(config, "scheduler_type", "polynomial")).lower()
    if sched_type == "ko":
        if KOConvexScheduler is None:
            raise RuntimeError("KOConvexScheduler not importable, but scheduler_type=ko.")
        return MixtureDiscreteProbPath(scheduler=KOConvexScheduler())
    return MixtureDiscreteProbPath(scheduler=PolynomialConvexScheduler(n=1.0))


def total_vocab_size(config) -> int:
    return int(config.datasets.audio_vocab_size) + int(config.model.audio_add_token)


def tsr_ratio_from_alpha(alpha_like: torch.Tensor, k: float, sigma: float, eps: float = 1e-12) -> torch.Tensor:
    """
    TSR ratio:
        ratio = (1 - a + a*sigma^2) / (1 - a + a*sigma^2/k)
    Treat scheduler.alpha_t as alpha-like signal power.
    """
    k = float(k)
    sigma = float(sigma)
    if k <= 0.0:
        raise ValueError("TSR k must be > 0.")
    a = alpha_like.clamp(0.0, 1.0)
    num = (1.0 - a) + a * (sigma * sigma)
    den = (1.0 - a) + a * (sigma * sigma / k)
    return (num / den.clamp_min(eps)).clamp_min(eps)


def compute_sigma_remdm(
    alpha_t: torch.Tensor,
    alpha_s: torch.Tensor,
    *,
    eta_rescale: float,
    eta_cap: float,
    tswitch: float,
    t_lin: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    ReMDM constraint:
        sigma_max = min(1, (1 - alpha_s) / alpha_t)
        sigma_t = eta_rescale * min(eta_cap, sigma_max)
    Optional switch: sigma=0 for t < tswitch.
    """
    one = torch.ones_like(alpha_t)
    alpha_t_safe = alpha_t.clamp_min(eps)
    sigma_max = torch.minimum(one, (1.0 - alpha_s).clamp_min(0.0) / alpha_t_safe)
    sigma_cap = torch.minimum(torch.full_like(alpha_t, float(eta_cap)), sigma_max)
    sigma = float(eta_rescale) * sigma_cap
    sigma = sigma.clamp(0.0, 1.0 - 1e-6)
    if float(t_lin) < float(tswitch):
        sigma = sigma * 0.0
    return sigma


# ----------------------------
# Correct CTMC sampler (mask-source) + optional PFG + TSR + ReMDM remasking
# ----------------------------

@torch.inference_mode()
def sample_mask_ctmc(
    *,
    config,
    model,
    path: MixtureDiscreteProbPath,
    text_ids: torch.Tensor,
    text_att_mask: torch.Tensor,
    codes_ref_1d: torch.Tensor,
    suffix_len: int,
    steps: int,
    device: torch.device,
    # sampling knobs
    x1_temp: float = 1.0,
    temp_schedule: str = "constant",   # "dfm36" or "constant"
    remask_noise: float = 0.0,         # old ad-hoc remasking
    # PFG
    use_pfg: bool = False,
    gamma: float = 1.0,
    # TSR
    use_tsr: bool = False,
    tsr_k: float = 1.0,
    tsr_sigma: float = 0.93,
    # ReMDM
    use_remdm: bool = False,
    remdm_eta_rescale: float = 0.3,
    remdm_eta_cap: float = 0.5,
    remdm_tswitch: float = 0.7,
    remdm_use_conf: bool = False,
    remdm_conf_threshold: float = 0.35,
    remdm_beta: float = 2.0,
    remdm_strength: float = 1.0,
    # length / EOS behavior
    use_oracle_length: bool = False,
    pin_final_eos: bool = True,
    allow_eos_remask: bool = False,
) -> torch.Tensor:
    """
    Returns full sequence including prefix: [1, prefix_len + suffix_len]
    """
    S = total_vocab_size(config)

    mask_id = int(config.datasets.audio_mask_token)
    eos_id = int(getattr(config.datasets, "audio_eos_token", -1))
    pad_id = getattr(config.datasets, "audio_pad_token", None)
    pad_id = int(pad_id) if pad_id is not None else None

    prefix_len = int(codes_ref_1d.numel())
    Ttot = prefix_len + int(suffix_len)

    xt = torch.full((1, Ttot), mask_id, device=device, dtype=torch.long)
    xt[:, :prefix_len] = codes_ref_1d.unsqueeze(0)

    # If using oracle length, you may want to keep a fixed EOS at the end (old behavior).
    # If you want to TEST EOS timing inference, pass pin_final_eos=False.
    if use_oracle_length and pin_final_eos:
        xt[0, -1] = eos_id

    audio_att_mask = torch.ones_like(xt, dtype=torch.bool, device=device)

    dt = 1.0 / max(1, int(steps))
    eps = 1e-12

    def temp_at(t_lin: float) -> float:
        if temp_schedule == "dfm36":
            return max(1e-3, float(x1_temp) * (1.0 - float(t_lin)) ** 2)
        return max(1e-3, float(x1_temp))

    if use_pfg:
        cond_drop_prob = float(getattr(config.datasets, "cond_drop_prob", 0.0))
        if cond_drop_prob <= 0.0:
            raise RuntimeError(
                "use_pfg=True but config.datasets.cond_drop_prob==0.0. "
                "This checkpoint likely never learned the unconditional branch."
            )

    for k in range(int(steps)):
        t_lin = k * dt
        t = torch.full((1,), float(t_lin), device=device, dtype=torch.float32)

        # also compute alpha at next time for ReMDM sigma_max constraint
        t_next_lin = min(1.0, (k + 1) * dt)
        t_next = torch.full((1,), float(t_next_lin), device=device, dtype=torch.float32)

        sched_t = path.scheduler(t)
        alpha_t = sched_t.alpha_t  # [1]
        dalpha_t = sched_t.d_alpha_t.clamp_min(1e-6)  # [1]

        sched_s = path.scheduler(t_next)
        alpha_s = sched_s.alpha_t  # [1]

        # lambda(t) = d alpha / (1 - alpha)
        lam = (dalpha_t / (1.0 - alpha_t).clamp_min(1e-6))  # [1]
        lam_scalar = float(lam.item())

        Tsoft = temp_at(t_lin)

        # TSR ratio (time-dependent scaling)
        if use_tsr and float(tsr_k) != 1.0:
            tsr_ratio = tsr_ratio_from_alpha(alpha_t, tsr_k, tsr_sigma)  # [1]
        else:
            tsr_ratio = None

        # conditional logits
        logits_c = model(
            x_t=xt,
            text_ids=text_ids,
            time=t,
            drop_text=False,
            text_att_mask=text_att_mask,
            audio_att_mask=audio_att_mask,
        ).float()

        # TSR on conditional logits
        if tsr_ratio is not None:
            logits_c = logits_c * tsr_ratio.view(1, 1, 1)

        probs_c = torch.softmax(logits_c / Tsoft, dim=-1)

        # unconditional for PFG
        if use_pfg:
            logits_u = model(
                x_t=xt,
                text_ids=text_ids,
                time=t,
                drop_text=True,
                text_att_mask=text_att_mask,
                audio_att_mask=audio_att_mask,
            ).float()

            # TSR on unconditional logits too (keep guidance consistent)
            if tsr_ratio is not None:
                logits_u = logits_u * tsr_ratio.view(1, 1, 1)

            probs_u = torch.softmax(logits_u / Tsoft, dim=-1)
        else:
            probs_u = None

        # Avoid sampling MASK as a target token
        probs_c[..., mask_id] = 0.0
        probs_c = probs_c / probs_c.sum(dim=-1, keepdim=True).clamp_min(eps)
        if probs_u is not None:
            probs_u[..., mask_id] = 0.0
            probs_u = probs_u / probs_u.sum(dim=-1, keepdim=True).clamp_min(eps)

        # Base unmask rates: mask-source => only MASK positions unmask
        xt_is_mask = (xt == mask_id).unsqueeze(-1).float()  # [1,T,1]
        base_r = (1.0 + float(remask_noise) * float(t_lin)) * lam_scalar  # scalar
        R_c = xt_is_mask * probs_c * base_r  # [1,T,S]

        if use_pfg:
            R_u = xt_is_mask * probs_u * base_r
        else:
            R_u = None

        # Protected tokens (NEVER remask / change):
        # - prefix prompt tokens always protected
        # - optionally final EOS if pin_final_eos
        protected = torch.zeros_like(xt, dtype=torch.bool, device=device)
        protected[:, :prefix_len] = True
        if pin_final_eos:
            protected[:, -1] = True

        # Old ad-hoc remasking noise (token -> MASK)
        # (Make sure it does NOT apply to protected tokens.)
        if remask_noise > 0.0:
            mask_one_hot = torch.zeros((S,), device=device, dtype=R_c.dtype)
            mask_one_hot[mask_id] = 1.0

            # only non-mask, non-protected positions get token->MASK noise
            eligible_noise = (~protected) & (xt != mask_id)

            R_c = R_c + eligible_noise.unsqueeze(-1).to(R_c.dtype) * mask_one_hot.view(1, 1, S) * float(remask_noise)
            if R_u is not None:
                R_u = R_u + eligible_noise.unsqueeze(-1).to(R_u.dtype) * mask_one_hot.view(1, 1, S) * float(remask_noise)

        # PFG mixing in log-rate space
        if use_pfg:
            logRc = torch.log(R_c + 1e-9)
            logRu = torch.log(R_u + 1e-9)
            R_mix = torch.exp(float(gamma) * logRc + (1.0 - float(gamma)) * logRu)
        else:
            R_mix = R_c

        # ----------------------------
        # ReMDM remasking (principled token -> MASK)
        # ----------------------------
        if use_remdm:
            sigma = compute_sigma_remdm(
                alpha_t=alpha_t,
                alpha_s=alpha_s,
                eta_rescale=float(remdm_eta_rescale),
                eta_cap=float(remdm_eta_cap),
                tswitch=float(remdm_tswitch),
                t_lin=float(t_lin),
            )  # [1]

            sigma_scalar = float(sigma.item())
            if sigma_scalar > 0.0:
                # Convert per-step remask probability sigma into a CTMC rate for this dt:
                #   p = 1 - exp(-dt * r) = sigma  =>  r = -log(1 - sigma) / dt
                r_base = -math.log(max(1e-9, 1.0 - sigma_scalar)) / max(1e-12, dt)

                # eligible positions:
                # - NOT protected (prefix, and final EOS if pinned)
                # - NOT MASK
                # - optionally: do NOT remask EOS tokens (recommended for stable EOS timing)
                eligible = (~protected) & (xt != mask_id)

                if not allow_eos_remask:
                    eligible = eligible & (xt != eos_id)

                # Optional confidence weighting (remask low-confidence tokens more)
                if remdm_use_conf:
                    if use_pfg:
                        # geometric mixture consistent with log-rate PFG
                        logp_c = torch.log(probs_c + eps)
                        logp_u = torch.log(probs_u + eps)
                        logp_mix = float(gamma) * logp_c + (1.0 - float(gamma)) * logp_u
                        probs_mix = torch.exp(logp_mix)
                        probs_mix = probs_mix / probs_mix.sum(dim=-1, keepdim=True).clamp_min(eps)
                    else:
                        probs_mix = probs_c

                    # current token prob under mixed probs
                    p_cur = probs_mix.gather(-1, xt[..., None].clamp(0, S - 1)).squeeze(-1)  # [1,T]
                    thr = float(remdm_conf_threshold)
                    beta = float(remdm_beta)

                    # weight in [0,1] where 1 = very low confidence, 0 = confident
                    low = (p_cur < thr).float()
                    w = low * ((thr - p_cur) / max(1e-9, thr)).clamp(0.0, 1.0) ** beta
                    w = (w * float(remdm_strength)).clamp(0.0, 1.0)
                else:
                    w = torch.ones_like(xt, dtype=torch.float32, device=device)

                # Add token->MASK rate on eligible positions
                r_pos = (r_base * w).to(R_mix.dtype)  # [1,T]
                add = torch.zeros((1, Ttot, S), device=device, dtype=R_mix.dtype)
                add[..., mask_id] = r_pos
                R_mix = R_mix + add * eligible.unsqueeze(-1).to(R_mix.dtype)

        # ----------------------------
        # CTMC tau-leap step
        # ----------------------------

        # Remove diagonal
        R_off = R_mix.clone()
        R_off.scatter_(-1, xt[..., None], 0.0)

        hazard = R_off.sum(dim=-1)  # [1,T]
        p_jump = 1.0 - torch.exp(-dt * hazard)  # [1,T]
        do_jump = (torch.rand_like(p_jump) < p_jump)

        # Never jump on prefix prompt tokens
        do_jump[:, :prefix_len] = False
        # If we pin final EOS, also never jump there (fixed token)
        if pin_final_eos:
            do_jump[:, -1] = False

        if do_jump.any():
            hazard_safe = hazard.clamp_min(1e-9)
            q = R_off / hazard_safe.unsqueeze(-1)  # [1,T,S]

            q2 = q.view(-1, S)
            jump_idx = do_jump.view(-1).nonzero(as_tuple=False).squeeze(-1)

            q_jump = q2.index_select(0, jump_idx)
            q_jump = torch.nan_to_num(q_jump, nan=0.0, posinf=0.0, neginf=0.0)
            q_jump = q_jump.clamp_min(0.0)
            q_jump = q_jump / q_jump.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            sampled = torch.multinomial(q_jump, 1).squeeze(-1)

            xt_flat = xt.view(-1)
            xt_flat[jump_idx] = sampled.to(xt_flat.dtype)
            xt = xt_flat.view_as(xt)

        # Keep prefix pinned ALWAYS
        xt[:, :prefix_len] = codes_ref_1d.unsqueeze(0)
        # Keep final EOS pinned ONLY if requested
        if pin_final_eos:
            xt[0, -1] = eos_id

        # Early stop only if NO remasking can happen
        # (ReMDM could still remask later and fix errors, especially with tswitch)
        if (remask_noise <= 0.0) and (not use_remdm):
            if (xt[:, prefix_len:] == mask_id).sum().item() == 0:
                break

    return xt


def load_audio(filepath: str, target_sr: int = 16_000) -> Tuple[torch.Tensor, int]:
    y, sr = torchaudio.load(filepath)
    if sr != target_sr:
        y = T.Resample(sr, target_sr)(y)

    if y.dim() == 1:
        y = y.unsqueeze(0).unsqueeze(0)
    if y.dim() == 2:
        y = y.unsqueeze(0)

    return y, target_sr


@torch.inference_mode()
def extract_codes(model: NeuCodec, filepath: str) -> torch.Tensor:
    y, sr = load_audio(filepath)
    with torch.no_grad():
        fsq_codes = model.encode_code(y)
    return fsq_codes.squeeze(0).cpu()


def load_codes(filepath: str) -> torch.Tensor:
    data = load_file(filepath)
    return data["fsq_codes"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Lightning .ckpt.")
    parser.add_argument("--metadata_csv", type=str, required=True,
                        help="CSV with columns: text, ref_text, filepath_codec, reference_codec")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--nsf", type=str, default="256",
                        help="Comma-separated steps, e.g. '4,8,16,32' or '256'")
    parser.add_argument("--use_oracle_length", action="store_true",
                        help="Use oracle token length from filepath_codec.")
    parser.add_argument("--oracle_add_eos", action="store_true",
                        help="If oracle tokens do NOT include EOS, add +1 length (optional).")

    # Sampling knobs
    parser.add_argument("--x1_temp", type=float, default=1.0)
    parser.add_argument("--temp_schedule", type=str, default="dfm36", choices=["dfm36", "constant"])
    parser.add_argument("--remask_noise", type=float, default=0.0,
                        help="Old ad-hoc remasking noise (token->MASK). Usually keep 0 if using --use_remdm.")

    # TSR
    parser.add_argument("--use_tsr", action="store_true", help="Enable Temporal Score Rescaling (TSR).")
    parser.add_argument("--tsr_k", type=float, default=1.0,
                        help="Sharpening factor k (>1 sharper, <1 flatter).")
    parser.add_argument("--tsr_sigma", type=float, default=0.1,
                        help="TSR sigma parameter.")

    # PFG
    parser.add_argument("--use_pfg", action="store_true")
    parser.add_argument("--gamma", type=float, default=2.5)

    # EOS behavior
    parser.add_argument("--no_pin_final_eos", action="store_true",
                        help="Disable forcing EOS at the final position. Use this to TEST EOS timing inference.")
    parser.add_argument("--allow_eos_remask", action="store_true",
                        help="Allow ReMDM to remask EOS tokens (NOT recommended for EOS timing evaluation).")

    # ReMDM remasking
    parser.add_argument("--use_remdm", action="store_true",
                        help="Enable ReMDM-style remasking with inference-time scaling.")
    parser.add_argument("--remdm_eta_rescale", type=float, default=0.3,
                        help="ReMDM rescale factor (multiplies min(eta_cap, sigma_max)).")
    parser.add_argument("--remdm_eta_cap", type=float, default=0.5,
                        help="ReMDM cap before rescale: min(eta_cap, sigma_max).")
    parser.add_argument("--remdm_tswitch", type=float, default=0.7,
                        help="Switch time: sigma=0 for t < tswitch (t in [0,1]). Set 0 for always-on.")
    parser.add_argument("--remdm_use_conf", action="store_true",
                        help="Enable confidence-based remasking (low-confidence tokens remask more).")
    parser.add_argument("--remdm_conf_threshold", type=float, default=0.35,
                        help="Confidence threshold: only tokens with p_cur < thr get remask weight.")
    parser.add_argument("--remdm_beta", type=float, default=2.0,
                        help="Sharpness of low-confidence weighting.")
    parser.add_argument("--remdm_strength", type=float, default=1.0,
                        help="Scale confidence weights (clamped to [0,1]).")

    parser.add_argument("--max_rows", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    seed_everything(args.seed)

    config = OmegaConf.load(args.config)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model = DFMTTSWrapper.load_from_checkpoint(
        args.checkpoint,
        config=config,
        map_location=device,
        strict=False,
        weights_only=False,
    ).to(device)
    model.eval()

    # Build path consistent with training
    path = build_path_from_config(config)

    # Load codec
    codec, saving_sr = load_codec(config, device)

    # Read metadata
    df = pd.read_csv(args.metadata_csv)
    if args.max_rows > 0:
        df = df.iloc[: args.max_rows].copy()

    steps_list = parse_int_list(args.nsf)

    for col in ["text", "ref_text", "filepath_codec", "reference_codec"]:
        if col not in df.columns:
            raise ValueError(f"metadata_csv missing required column: {col}")

    eos_id = int(getattr(config.datasets, "audio_eos_token", -1))
    mask_id = int(getattr(config.datasets, "audio_mask_token", -1))

    pin_final_eos = (not bool(args.no_pin_final_eos))
    allow_eos_remask = bool(args.allow_eos_remask)

    rtf_tuples = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            text = str(row["text"])
            text_ref = str(row["ref_text"]) if not pd.isna(row["ref_text"]) else None

            filepath_codec = str(row["filepath_codec"])
            filepath_codec = filepath_codec.replace("/xcodec2/LibriSpeech-test-clean-filtered/", "/neucodec/LibriSpeech/")
            filepath_codec = filepath_codec.replace(".pt", ".safetensors")

            ref_filepath_codec = str(row["reference_codec"])
            ref_filepath_codec = ref_filepath_codec.replace("/xcodec2/LibriSpeech-test-clean-filtered/", "/neucodec/LibriSpeech/")
            ref_filepath_codec = ref_filepath_codec.replace(".pt", ".safetensors")

            assert os.path.exists(filepath_codec), f"File not found: {filepath_codec}"
            assert os.path.exists(ref_filepath_codec), f"File not found: {ref_filepath_codec}"

            codes_ref = load_codes(ref_filepath_codec).squeeze()
            if codes_ref.ndim != 1:
                codes_ref = codes_ref.reshape(-1)
            codes_ref = codes_ref.long().to(device)

            # text inputs
            text_ids, text_att_mask, _tok = build_text_inputs(config, text, text_ref, device)

            # Oracle length
            oracle_len = None
            if args.use_oracle_length:
                oracle_codes = load_codes(filepath_codec).squeeze()
                if oracle_codes.ndim != 1:
                    oracle_codes = oracle_codes.reshape(-1)
                oracle_len = int(oracle_codes.numel())
                if args.oracle_add_eos:
                    oracle_len += 1

            suffix_len = oracle_len if oracle_len is not None else 2048

            print(f"Sufix length: {suffix_len} (oracle length: {oracle_len})")

            for steps in steps_list:
                out_wav = os.path.join(args.output_dir, f"audio_{idx}-nsf{steps}.wav")
                if os.path.exists(out_wav):
                    continue

                with torch.no_grad():
                    start_time = time.time()
                    xt_full = sample_mask_ctmc(
                        config=config,
                        model=model,
                        path=path,
                        text_ids=text_ids,
                        text_att_mask=text_att_mask,
                        codes_ref_1d=codes_ref,
                        suffix_len=suffix_len,
                        steps=steps,
                        device=device,
                        x1_temp=float(args.x1_temp),
                        temp_schedule=str(args.temp_schedule),
                        remask_noise=float(args.remask_noise),
                        use_pfg=bool(args.use_pfg),
                        gamma=float(args.gamma),
                        use_tsr=bool(args.use_tsr),
                        tsr_k=float(args.tsr_k),
                        tsr_sigma=float(args.tsr_sigma),
                        use_remdm=bool(args.use_remdm),
                        remdm_eta_rescale=float(args.remdm_eta_rescale),
                        remdm_eta_cap=float(args.remdm_eta_cap),
                        remdm_tswitch=float(args.remdm_tswitch),
                        remdm_use_conf=bool(args.remdm_use_conf),
                        remdm_conf_threshold=float(args.remdm_conf_threshold),
                        remdm_beta=float(args.remdm_beta),
                        remdm_strength=float(args.remdm_strength),
                        use_oracle_length=bool(args.use_oracle_length),
                        pin_final_eos=pin_final_eos,
                        allow_eos_remask=allow_eos_remask,
                    )
                    end_time = time.time()
                    total_pred_time = end_time - start_time

                prefix_len = int(codes_ref.numel())
                gen = xt_full[0, prefix_len:].detach().cpu()

                # Truncate at first EOS, then drop any remaining MASK
                gen = truncate_at_first_eos(gen, eos_id)
                gen = gen[gen != mask_id]

                gen_for_codec = gen.to(device).unsqueeze(0).unsqueeze(0)  # [1,1,T]
                wav = codec.decode_code(gen_for_codec).detach()

                total_wav_length = wav.shape[-1] / saving_sr
                rtf = total_pred_time / total_wav_length if total_wav_length > 0 else float("inf")
                rtf_tuples.append((idx, steps, total_pred_time, total_wav_length, rtf))

                torchaudio.save(out_wav, wav.squeeze(0).cpu(), saving_sr)

        except Exception as e:
            print(f"[row {idx}] error: {e}")
            continue

    rtf_df = pd.DataFrame(rtf_tuples, columns=["idx", "steps", "total_pred_time", "total_wav_length", "rtf"])
    rtf_df.to_csv(os.path.join(args.output_dir, "rtf_results.csv"), index=False)


if __name__ == "__main__":
    main()
