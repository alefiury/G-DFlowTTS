#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DFM-TTS inference (mask-source, MixtureDiscreteProbPath) with CTMC tau-leap sampler.

Includes:
- Correct CTMC jump probability: p_jump = 1 - exp(-dt * hazard)
- lambda(t) = d_alpha_t / (1 - alpha_t)
- Optional Predictor-Free Guidance (PFG) in log-rate space
- LLaDA-inspired *low-confidence remasking* implemented as CTMC rates:
    token -> MASK transitions are added only for low-confidence tokens,
    and scaled so expected remasks ~ "excess unmasked" vs a target schedule.

Notes:
- Model time input assumed linear t in [0,1].
- Pure mask-source has no remasking; remasking is optional noise/heuristic.
"""

import os
import time
import math
import argparse
import warnings
from typing import Optional, List, Tuple

warnings.filterwarnings("ignore")

import torch
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
    # e.g. "4,8,16,32" or "256"
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def ensure_gpt2_padding(tok: AutoTokenizer) -> AutoTokenizer:
    # GPT-2 has no pad token by default; training collator sets pad_token = eos_token
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": tok.eos_token})
    return tok


def build_text_inputs(config, sentence: str, text_ref: Optional[str], device: torch.device):
    """
    Match training:
    - If datasets.type == hf_text_tokenizer: use HF tokenizer with padding/attention_mask
    - Else: use VoiceBpeTokenizer (legacy)
    """
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
    """
    Returns: codec_model, saving_sr
    """
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
        # NeuCodec commonly decodes at 24kHz
        saving_sr = 24000
        return codec, saving_sr

    raise ValueError(f"Unsupported codec_name: {codec_name}")


def build_path_from_config(config):
    """
    Match training wrapper:
    - scheduler_type: "polynomial" (PolynomialConvexScheduler(n=1.0)) or "ko"
    """
    sched_type = str(getattr(config, "scheduler_type", "polynomial")).lower()
    if sched_type == "ko":
        if KOConvexScheduler is None:
            raise RuntimeError("KOConvexScheduler not importable, but scheduler_type=ko.")
        return MixtureDiscreteProbPath(scheduler=KOConvexScheduler())
    return MixtureDiscreteProbPath(scheduler=PolynomialConvexScheduler(n=1.0))


def total_vocab_size(config) -> int:
    # training uses: audio_vocab_size + model.audio_add_token
    return int(config.datasets.audio_vocab_size) + int(config.model.audio_add_token)


# ----------------------------
# Optional duration model
# ----------------------------

@torch.inference_mode()
def get_remaining_duration(
    duration_model: DurationPredictorWrapper,
    text_ids: torch.Tensor,
    codes_ref_1d: torch.Tensor,
    config,
    device: torch.device,
) -> int:
    """
    Returns an int length for the generated suffix.
    """
    if duration_model is None:
        raise RuntimeError("Duration model is None; cannot predict length.")

    bos_id = int(getattr(config.datasets, "audio_eos_token", 0))
    bos_vec = codes_ref_1d.new_full((1,), bos_id, dtype=torch.long)
    codes_ref = torch.cat((bos_vec, codes_ref_1d), dim=0)

    remaining = duration_model(
        text_ids=text_ids,
        audio_ids=codes_ref.unsqueeze(0).to(device),
    )
    return int(torch.argmax(remaining[:, -1], dim=-1).item())


# ----------------------------
# LLaDA-style remasking as CTMC rates
# ----------------------------

def _compute_confidence_probs(
    *,
    logits_c: torch.Tensor,
    logits_u: Optional[torch.Tensor],
    use_pfg: bool,
    gamma: float,
    temp_conf: float,
) -> torch.Tensor:
    """
    Confidence distribution used ONLY for computing confidence (not for sampling).
    Uses temp_conf (default 1.0) so confidence is not distorted by sampling temperature.
    If PFG: mixes in log-prob space then renormalizes.
    Returns probs_conf: [1,T,S]
    """
    eps = 1e-9
    Pc = torch.softmax(logits_c / float(temp_conf), dim=-1)
    if not use_pfg:
        return Pc

    assert logits_u is not None
    Pu = torch.softmax(logits_u / float(temp_conf), dim=-1)

    # log-prob mix then renormalize
    logPc = torch.log(Pc + eps)
    logPu = torch.log(Pu + eps)
    Pmix = torch.softmax(float(gamma) * logPc + (1.0 - float(gamma)) * logPu, dim=-1)
    return Pmix


def _llada_remask_rates_from_budget(
    *,
    xt: torch.Tensor,                 # [1,T]
    probs_conf: torch.Tensor,         # [1,T,S]
    prefix_len: int,
    mask_id: int,
    eos_pos: int,
    dt: float,
    step_idx: int,
    steps: int,
    # knobs
    remask_rate_max: float,
    conf_threshold: float,
    beta: float,
    strength: float,
) -> torch.Tensor:
    """
    Returns remask_rate per position: [1,T], to be added as rate into the MASK column.
    This is inspired by LLaDA low-confidence remasking, but done as CTMC rates:

    - Define a target #unmasked at next step with a linear schedule.
    - If we have "excess" unmasked tokens, allocate a per-token remask probability mass
      proportional to low-confidence weights, then convert to rates.

    We only consider suffix positions excluding the final EOS slot.
    """
    device = xt.device
    B, T = xt.shape
    assert B == 1

    remask_rate = torch.zeros((1, T), device=device, dtype=torch.float32)
    if remask_rate_max <= 0.0 or strength <= 0.0:
        return remask_rate

    # Suffix region excluding EOS slot
    suf_start = prefix_len
    suf_end = eos_pos  # exclude eos_pos itself
    if suf_end <= suf_start:
        return remask_rate

    # current-token confidence c_i = p(x_i)
    p_cur = probs_conf.gather(-1, xt[..., None]).squeeze(-1)  # [1,T]

    # candidates: suffix positions that are currently unmasked (not MASK), excluding EOS slot
    cand = torch.zeros_like(p_cur, dtype=torch.bool)
    cand[:, suf_start:suf_end] = True
    cand = cand & (xt != mask_id)

    num_cand = int(cand.sum().item())
    if num_cand == 0:
        return remask_rate

    # LLaDA-like target schedule on suffix (excluding EOS slot):
    # start: 0 unmasked, end: all unmasked. Use t_next = (k+1)/steps.
    t_next = float(step_idx + 1) / float(max(1, steps))
    L = (suf_end - suf_start)
    target_unmasked = int(round(L * t_next))

    cur_unmasked = int((xt[:, suf_start:suf_end] != mask_id).sum().item())
    excess = cur_unmasked - target_unmasked

    if excess <= 0:
        return remask_rate

    # low-confidence weights (only below a threshold)
    thr = max(1e-6, float(conf_threshold))
    bad = ((thr - p_cur).clamp_min(0.0) / thr) ** float(beta)  # [1,T]
    w = bad[cand]  # [Nc]

    wsum = float(w.sum().item())
    if wsum <= 0.0:
        # if everything is above threshold, don't remask (LLADA wouldn't either)
        return remask_rate

    # We want expected remasks this step ~= excess * strength (bounded).
    desired = int(min(num_cand, max(0, int(round(excess * float(strength))))))
    if desired <= 0:
        return remask_rate

    # Allocate per-candidate remask probabilities p_i such that sum p_i = desired.
    # p_i in (0,1), then rate_i = -log(1 - p_i) / dt, capped by remask_rate_max.
    w_norm = w / wsum
    p_i = float(desired) * w_norm  # [Nc]
    p_i = torch.clamp(p_i, 0.0, 0.999)  # avoid inf
    rate_i = -torch.log1p(-p_i) / float(max(1e-9, dt))
    rate_i = torch.clamp(rate_i, 0.0, float(remask_rate_max))

    # write back to [1,T]
    remask_rate[cand] = rate_i.to(remask_rate.dtype)
    return remask_rate


# ----------------------------
# Correct CTMC sampler (mask-source) + optional PFG + optional remask-as-rates
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
    use_pfg: bool = False,
    gamma: float = 1.0,               # in log-rate space; 1=conditional, 0=unconditional
    # remasking knobs (rates)
    remask_mode: str = "llada",       # "none" | "flat" | "llada"
    remask_rate_max: float = 0.0,     # max rate for token->MASK
    remask_conf_temp: float = 1.0,    # temp for confidence only
    remask_conf_threshold: float = 0.35,
    remask_beta: float = 2.0,
    remask_strength: float = 1.0,
) -> torch.Tensor:
    """
    Returns full sequence including prefix: [1, prefix_len + suffix_len]
    """
    S = total_vocab_size(config)

    mask_id = int(config.datasets.audio_mask_token)
    eos_id = int(getattr(config.datasets, "audio_eos_token", -1))

    prefix_len = int(codes_ref_1d.numel())
    Ttotal = prefix_len + int(suffix_len)
    eos_pos = Ttotal - 1

    # init all MASK, then pin prefix
    xt = torch.full((1, Ttotal), mask_id, device=device, dtype=torch.long)
    xt[:, :prefix_len] = codes_ref_1d.unsqueeze(0)

    # pin EOS at the end (always)
    xt[0, eos_pos] = eos_id

    audio_att_mask = torch.ones_like(xt, dtype=torch.bool, device=device)

    dt = 1.0 / max(1, int(steps))
    eps = 1e-12

    def temp_at(t_lin: float) -> float:
        if temp_schedule == "dfm36":
            return max(1e-3, float(x1_temp) * (1.0 - float(t_lin)) ** 2)
        return max(1e-3, float(x1_temp))

    # sanity: PFG only if the model was trained with conditional drop
    if use_pfg:
        cond_drop_prob = float(getattr(config.datasets, "cond_drop_prob", 0.0))
        if cond_drop_prob <= 0.0:
            raise RuntimeError(
                "use_pfg=True but config.datasets.cond_drop_prob==0.0. "
                "This checkpoint likely never learned the unconditional branch."
            )

    remask_mode = str(remask_mode).lower()

    for k in range(int(steps)):
        t_lin = k * dt
        t = torch.full((1,), float(t_lin), device=device, dtype=torch.float32)

        sched = path.scheduler(t)
        alpha_t = sched.alpha_t
        dalpha_t = sched.d_alpha_t.clamp_min(1e-6)

        # lambda(t) = d alpha / (1 - alpha)
        lam = (dalpha_t / (1.0 - alpha_t).clamp_min(1e-6))
        lam_scalar = float(lam.item())

        # sampling temperature
        Tsoft = temp_at(t_lin)

        # conditional logits
        logits_c = model(
            x_t=xt,
            text_ids=text_ids,
            time=t,
            drop_text=False,
            text_att_mask=text_att_mask,
            audio_att_mask=audio_att_mask,
        ).float()

        probs_c = torch.softmax(logits_c / Tsoft, dim=-1)

        if use_pfg:
            logits_u = model(
                x_t=xt,
                text_ids=text_ids,
                time=t,
                drop_text=True,
                text_att_mask=text_att_mask,
                audio_att_mask=audio_att_mask,
            ).float()
            probs_u = torch.softmax(logits_u / Tsoft, dim=-1)
        else:
            logits_u = None
            probs_u = None

        # never sample MASK as target token
        probs_c[..., mask_id] = 0.0
        probs_c = probs_c / probs_c.sum(dim=-1, keepdim=True).clamp_min(eps)
        if probs_u is not None:
            probs_u[..., mask_id] = 0.0
            probs_u = probs_u / probs_u.sum(dim=-1, keepdim=True).clamp_min(eps)

        # mask-source unmask rates: only MASK positions can jump to vocab with rate = lam * p(token)
        xt_is_mask = (xt == mask_id).unsqueeze(-1).float()  # [1,T,1]
        base_r = lam_scalar  # keep unmask dynamics clean; remask handled separately

        R_c = xt_is_mask * probs_c * base_r  # [1,T,S]
        R_u = (xt_is_mask * probs_u * base_r) if (probs_u is not None) else None

        # --- remask as rates (token -> MASK) ---
        if remask_mode != "none" and remask_rate_max > 0.0:
            if remask_mode == "flat":
                # constant rate for any unmasked suffix position (excluding prefix/EOS)
                remask_rate = torch.zeros((1, Ttotal), device=device, dtype=torch.float32)
                remask_rate[:, prefix_len:eos_pos] = float(remask_rate_max)
                remask_rate = remask_rate * (xt != mask_id).float()
            elif remask_mode == "llada":
                probs_conf = _compute_confidence_probs(
                    logits_c=logits_c,
                    logits_u=logits_u,
                    use_pfg=use_pfg,
                    gamma=float(gamma),
                    temp_conf=float(remask_conf_temp),
                )
                remask_rate = _llada_remask_rates_from_budget(
                    xt=xt,
                    probs_conf=probs_conf,
                    prefix_len=prefix_len,
                    mask_id=mask_id,
                    eos_pos=eos_pos,
                    dt=dt,
                    step_idx=k,
                    steps=int(steps),
                    remask_rate_max=float(remask_rate_max),
                    conf_threshold=float(remask_conf_threshold),
                    beta=float(remask_beta),
                    strength=float(remask_strength),
                )
            else:
                raise ValueError(f"Unknown remask_mode: {remask_mode}")

            # add token->MASK rate into the MASK column (for both branches so PFG mixing preserves it)
            R_c = R_c.clone()
            R_c[..., mask_id] += remask_rate
            if R_u is not None:
                R_u = R_u.clone()
                R_u[..., mask_id] += remask_rate

        # PFG mixing in log-rate space: R_mix ∝ R_c^gamma * R_u^(1-gamma)
        if use_pfg:
            logRc = torch.log(R_c + 1e-9)
            logRu = torch.log(R_u + 1e-9)
            R_mix = torch.exp(float(gamma) * logRc + (1.0 - float(gamma)) * logRu)
        else:
            R_mix = R_c

        # Remove diagonal (no self-jumps)
        R_off = R_mix.clone()
        R_off.scatter_(-1, xt[..., None], 0.0)

        # Total hazard per position
        hazard = R_off.sum(dim=-1)  # [1,T]

        # Jump probability per position
        p_jump = 1.0 - torch.exp(-dt * hazard)  # [1,T]
        do_jump = (torch.rand_like(p_jump) < p_jump)

        # Never jump on pinned prefix or pinned EOS
        do_jump[:, :prefix_len] = False
        do_jump[:, eos_pos] = False

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

        # Early stop: if suffix (excluding EOS slot) has no MASK and remask is off
        if remask_mode == "none":
            if (xt[:, prefix_len:eos_pos] == mask_id).sum().item() == 0:
                break

        # re-pin prefix and eos
        xt[:, :prefix_len] = codes_ref_1d.unsqueeze(0)
        xt[0, eos_pos] = eos_id

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
    y, _sr = load_audio(filepath)
    fsq_codes = model.encode_code(y)
    return fsq_codes.squeeze(0).cpu()


def load_codes(filepath: str) -> torch.Tensor:
    data = load_file(filepath)
    return data["fsq_codes"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Lightning .ckpt.")
    parser.add_argument("--metadata_csv", type=str, required=True, help="CSV columns: text, ref_text, filepath_codec, reference_codec")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--nsf", type=str, default="256", help="Comma-separated steps, e.g. '4,8,16,32' or '256'")
    parser.add_argument("--use_oracle_length", action="store_true", help="Use oracle token length from filepath_codec.")
    parser.add_argument("--oracle_add_eos", action="store_true", help="If oracle tokens do NOT include EOS, add +1 length.")

    # Duration model (optional)
    parser.add_argument("--duration_config", type=str, default=None)
    parser.add_argument("--duration_ckpt", type=str, default=None)

    # Sampling knobs
    parser.add_argument("--x1_temp", type=float, default=1.0)
    parser.add_argument("--temp_schedule", type=str, default="dfm36", choices=["dfm36", "constant"])

    # PFG
    parser.add_argument("--use_pfg", action="store_true")
    parser.add_argument("--gamma", type=float, default=2.0)

    # Remasking (rates)
    parser.add_argument("--remask_mode", type=str, default="llada", choices=["none", "flat", "llada"])
    parser.add_argument("--remask_rate_max", type=float, default=0.5,
                        help="Max CTMC rate for token->MASK. 0 disables remasking.")
    parser.add_argument("--remask_conf_temp", type=float, default=1.0,
                        help="Temperature used ONLY for confidence computation.")
    parser.add_argument("--remask_conf_threshold", type=float, default=0.35,
                        help="Only tokens with p_cur < threshold get remask weight.")
    parser.add_argument("--remask_beta", type=float, default=2.0,
                        help="Sharpness of low-confidence weighting.")
    parser.add_argument("--remask_strength", type=float, default=1.0,
                        help="Scale desired remasks (1.0 ~ match excess budget).")

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

    # Optional duration model
    duration_model = None
    if not args.use_oracle_length and args.duration_config and args.duration_ckpt:
        dur_cfg = OmegaConf.load(args.duration_config)
        duration_model = DurationPredictorWrapper.load_from_checkpoint(
            args.duration_ckpt,
            config=dur_cfg,
            map_location=device,
            strict=False,
        ).to(device)
        duration_model.eval()

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

            # Text inputs (same tokenizer as training)
            text_ids, text_att_mask, _tok = build_text_inputs(config, text, text_ref, device)

            # Oracle suffix length
            oracle_len = None
            if args.use_oracle_length:
                oracle_codes = load_codes(filepath_codec).squeeze()
                if oracle_codes.ndim != 1:
                    oracle_codes = oracle_codes.reshape(-1)
                oracle_len = int(oracle_codes.numel())
                if args.oracle_add_eos:
                    oracle_len += 1

            if oracle_len is None:
                suffix_len = 2048
            else:
                suffix_len = oracle_len

            for steps in steps_list:
                out_wav = os.path.join(args.output_dir, f"audio_{idx}-nsf{steps}.wav")
                if os.path.exists(out_wav):
                    continue

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
                    use_pfg=bool(args.use_pfg),
                    gamma=float(args.gamma),
                    remask_mode=str(args.remask_mode),
                    remask_rate_max=float(args.remask_rate_max),
                    remask_conf_temp=float(args.remask_conf_temp),
                    remask_conf_threshold=float(args.remask_conf_threshold),
                    remask_beta=float(args.remask_beta),
                    remask_strength=float(args.remask_strength),
                )
                total_pred_time = time.time() - start_time

                # Extract generated suffix
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
