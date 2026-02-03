#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DFM-TTS inference (mask-source, MixtureDiscreteProbPath) — aligned with:
- Flow Matching Guide/Code discrete CTMC sampler (jump-probability tau-leap)
- Your training wrapper: model time input is linear t in [0,1]
- Your config: hf_text_tokenizer (GPT-2), mask source, polynomial scheduler
- Optional predictor-free guidance (PFG) ONLY if the checkpoint was trained with cond_drop_prob > 0

Key fixes vs your current script:
- Uses the SAME tokenizer as training (HF AutoTokenizer if datasets.type == hf_text_tokenizer)
- Correct CTMC step: jump prob = 1 - exp(-dt * hazard), hazard = sum(off-diag rates)
- Uses scheduler alpha_t and d_alpha_t: lambda(t) = d_alpha_t / (1 - alpha_t)
- Proper mask-absorbing dynamics: only MASK positions unmask (unless remask noise is enabled)
- Safe EOS handling: truncate at first EOS (don’t delete all EOS blindly)
"""

import os
import math
import argparse
import warnings
from dataclasses import dataclass
from typing import Optional, List, Tuple

warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import pandas as pd
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
# Length (optional duration model)
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
    Your original helper, but made config-aware.
    Returns an int length for the generated suffix.
    """
    if duration_model is None:
        raise RuntimeError("Duration model is None; cannot predict length.")

    # In your earlier code you used 65536 as "bos". Use audio_eos_token as a safe BOS-like marker.
    bos_id = int(getattr(config.datasets, "audio_eos_token", 0))
    bos_vec = codes_ref_1d.new_full((1,), bos_id, dtype=torch.long)
    codes_ref = torch.cat((bos_vec, codes_ref_1d), dim=0)  # [prefix+1]

    remaining = duration_model(
        text_ids=text_ids,
        audio_ids=codes_ref.unsqueeze(0).to(device),
    )
    # last timestep distribution -> argmax
    return int(torch.argmax(remaining[:, -1], dim=-1).item())


def ramp_up(t, t_start=0.5, t_end=1.0):
    # 0 until t_start, then linear -> 1 at t_end
    if t <= t_start:
        return 0.0
    if t >= t_end:
        return 1.0
    return (t - t_start) / (t_end - t_start)

def tsr_scale_from_kappa(kappa, k=1.0, eps=1e-6):
    # eta = kappa/(1-kappa), s = (eta+1)/(eta/k + 1)
    kappa = float(max(eps, min(1.0 - eps, kappa)))
    eta = kappa / (1.0 - kappa)
    return (eta + 1.0) / (eta / float(k) + 1.0)

@torch.inference_mode()
def sample_ctmc_refine_with_guidance(
    *,
    config,
    model,
    path,
    text_ids,
    text_att_mask,
    codes_ref_1d,
    suffix_len,
    steps,
    device,
    # guidance
    use_pfg=True,
    w_max=2.5,          # like CFG scale; >1 strengthens conditional
    w_start=0.0,        # start ramp at t=0.0
    w_end=1.0,
    # TSR-like temperature control
    tsr_k=2.0,          # >1 => sharper (lower temp late), <1 => flatter
    # misc
    pin_eos=True,
):
    S = int(config.datasets.audio_vocab_size) + int(config.model.audio_add_token)
    mask_id = int(config.datasets.audio_mask_token)
    eos_id  = int(getattr(config.datasets, "audio_eos_token", -1))

    prefix_len = int(codes_ref_1d.numel())
    T = prefix_len + int(suffix_len)

    xt = torch.full((1, T), mask_id, device=device, dtype=torch.long)
    xt[:, :prefix_len] = codes_ref_1d.unsqueeze(0)

    # If you want EOS pinned, put it and never allow it to jump.
    if eos_id >= 0:
        xt[0, -1] = eos_id

    audio_att_mask = torch.ones_like(xt, dtype=torch.bool, device=device)

    dt = 1.0 / max(1, int(steps))
    eps = 1e-12

    for k in range(int(steps)):
        t_lin = k * dt
        t = torch.full((1,), float(t_lin), device=device, dtype=torch.float32)

        sched = path.scheduler(t)
        kappa_t = float(sched.alpha_t.item())
        dkappa_t = float(max(1e-6, sched.d_alpha_t.item()))

        # lambda(t) = dkappa/(1-kappa)
        lam = dkappa_t / max(1e-6, (1.0 - kappa_t))

        # ---- model calls (cond / uncond) ----
        logits_c = model(
            x_t=xt,
            text_ids=text_ids,
            time=t,
            drop_text=False,
            text_att_mask=text_att_mask,
            audio_att_mask=audio_att_mask,
        ).float()

        if use_pfg:
            logits_u = model(
                x_t=xt,
                text_ids=text_ids,
                time=t,
                drop_text=True,
                text_att_mask=text_att_mask,
                audio_att_mask=audio_att_mask,
            ).float()
        else:
            logits_u = None

        # ---- convert to log-probs ----
        logp_c = torch.log_softmax(logits_c, dim=-1)

        if logits_u is not None:
            logp_u = torch.log_softmax(logits_u, dim=-1)

            # ramp-up schedule: low guidance early, higher late
            g = w_max * ramp_up(float(t_lin), t_start=w_start, t_end=w_end)

            # normalized “CFG/PFG in log-prob space”
            # log p_guided ∝ (1-g)*logp_u + g*logp_c
            logp = (1.0 - g) * logp_u + g * logp_c
        else:
            logp = logp_c

        # ---- TSR-like time-dependent scaling (acts like inverse temperature) ----
        s_t = tsr_scale_from_kappa(kappa_t, k=tsr_k)
        logp = logp * float(s_t)

        # ---- back to probabilities ----
        p_theta = torch.softmax(logp, dim=-1)

        # never propose MASK as a target token (optional but usually good)
        p_theta[..., mask_id] = 0.0
        p_theta = p_theta / p_theta.sum(dim=-1, keepdim=True).clamp_min(eps)

        # ---- DFM Euler CTMC rates ----
        # u(x|current) = lam * p_theta(x), then remove diagonal
        u = lam * p_theta
        u.scatter_(-1, xt[..., None], 0.0)

        # hazard per position
        hazard = u.sum(dim=-1)  # [1,T]

        # jump decision
        p_jump = 1.0 - torch.exp(-dt * hazard)
        do_jump = (torch.rand_like(p_jump) < p_jump)

        # pin prefix
        do_jump[:, :prefix_len] = False

        # optionally pin EOS at last position
        if pin_eos and eos_id >= 0:
            do_jump[:, -1] = False

        if do_jump.any():
            hazard_safe = hazard.clamp_min(1e-9)
            q = u / hazard_safe.unsqueeze(-1)

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

    return xt


def load_audio(filepath: str, target_sr: int = 16_000) -> Tuple[torch.Tensor, int]:
    """
    Load audio file and resample to target sample rate if necessary.

    Args:
        filepath (str): Path to the audio file.
        target_sr (int): Target sample rate. Default is 16,000 Hz.

    Returns:
        Tuple[torch.Tensor, int]: Loaded audio tensor and the sample rate.
    """
    y, sr = torchaudio.load(filepath)  # (1, T)
    if sr != target_sr:
        y = T.Resample(sr, target_sr)(y)

    if y.dim() == 1:
        y = y.unsqueeze(0).unsqueeze(0)
    # put in # (1, 1, T_16) if not
    if y.dim() == 2:
        y = y.unsqueeze(0)

    return y, target_sr  # (1, 1, T_16)


@torch.inference_mode()
def extract_codes(model: NeuCodec, filepath: str) -> torch.Tensor:
    """
    Extract feature codes from audio file.

    Args:
        model (NeuCodec): Pretrained NeuCodec model.
        filepath (str): Path to the audio file.
    Returns:
        torch.Tensor: Extracted feature codes.
    """
    y, sr = load_audio(filepath)  # (1, 1, T_16)
    with torch.no_grad():
        fsq_codes = model.encode_code(y)  # (1, T_code)

    return fsq_codes.squeeze(0).cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Lightning .ckpt.")
    parser.add_argument("--metadata_csv", type=str, required=True, help="CSV with columns: text, ref_text, filepath_codec, reference_codec")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--nsf", type=str, default="256", help="Comma-separated steps, e.g. '4,8,16,32' or '256'")
    parser.add_argument("--use_oracle_length", action="store_true", help="Use oracle token length from filepath_codec.")
    parser.add_argument("--oracle_add_eos", action="store_true", help="If oracle tokens do NOT include EOS, add +1 length (optional).")

    # Duration model (optional)
    parser.add_argument("--duration_config", type=str, default=None)
    parser.add_argument("--duration_ckpt", type=str, default=None)

    # Sampling knobs
    parser.add_argument("--x1_temp", type=float, default=1.0)
    parser.add_argument("--temp_schedule", type=str, default="dfm36", choices=["dfm36", "constant"])
    parser.add_argument("--remask_noise", type=float, default=0.0)

    # PFG (only for checkpoints trained with cond_drop_prob > 0)
    parser.add_argument("--use_pfg", action="store_true")
    parser.add_argument("--gamma", type=float, default=2.5)

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
    if not args.use_oracle_length:
        if args.duration_config is None or args.duration_ckpt is None:
            pass
        else:
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

    # Required columns
    for col in ["text", "ref_text", "filepath_codec", "reference_codec"]:
        if col not in df.columns:
            raise ValueError(f"metadata_csv missing required column: {col}")

    # Token ids
    eos_id = int(getattr(config.datasets, "audio_eos_token", -1))
    mask_id = int(getattr(config.datasets, "audio_mask_token", -1))

    for idx, row in df.iterrows():
        # try:
        text = str(row["text"])
        text_ref = str(row["ref_text"]) if not pd.isna(row["ref_text"]) else None

        filepath_codec = str(row["filepath_codec"])
        ref_filepath_codec = str(row["reference_codec"])

        target_filepath = str(row["filepath"])
        reference_filepath = str(row["reference"])

        # Load reference codes (prefix)
        # codes_ref = torch.load(ref_filepath_codec).squeeze()
        codes_ref = extract_codes(codec, reference_filepath).squeeze()
        if codes_ref.ndim != 1:
            codes_ref = codes_ref.reshape(-1)
        codes_ref = codes_ref.long().to(device)

        # Build text inputs using the SAME tokenizer as training
        text_ids, text_att_mask, _tok = build_text_inputs(config, text, text_ref, device)

        # Oracle length (suffix length)
        oracle_len = None
        if args.use_oracle_length:
            # oracle_codes = torch.load(filepath_codec).squeeze()
            oracle_codes = extract_codes(codec, target_filepath).squeeze()
            if oracle_codes.ndim != 1:
                oracle_codes = oracle_codes.reshape(-1)
            oracle_len = int(oracle_codes.numel())
            if args.oracle_add_eos:
                oracle_len += 1

        print(f"\n\n oracle_len: {oracle_len}-{oracle_len/50} \n\n")

        # Predict suffix length if needed
        if oracle_len is None:
            # suffix_len = get_remaining_duration(
            #     duration_model=duration_model,
            #     text_ids=text_ids,
            #     codes_ref_1d=codes_ref,
            #     config=config,
            #     device=device,
            # )
            suffix_len = 2048
            print(f"[row {idx}] using fixed suffix length: {suffix_len}")
        else:
            suffix_len = oracle_len

        # print(f"\n\n suffix_len:{suffix_len}-{suffix_len/50} \n\n")

        for steps in steps_list:
            # print(f"\n\n suffix_len:{suffix_len}  | {steps}\n\n")
            out_wav = os.path.join(args.output_dir, f"audio_{idx}-nsf{steps}.wav")
            if os.path.exists(out_wav):
                continue

            with torch.no_grad():
                xt_full = sample_ctmc_refine_with_guidance(
                    config=config,
                    model=model,
                    path=path,
                    text_ids=text_ids,
                    text_att_mask=text_att_mask,
                    codes_ref_1d=codes_ref,
                    suffix_len=suffix_len,
                    steps=steps,
                    device=device,
                    # x1_temp=float(args.x1_temp),
                    # temp_schedule=str(args.temp_schedule),
                    # remask_noise=float(args.remask_noise),
                    # use_pfg=bool(args.use_pfg),
                    # gamma=float(args.gamma),
                    use_pfg=bool(args.use_pfg),
                    w_max=float(args.gamma),
                )

                # Extract generated suffix
                prefix_len = int(codes_ref.numel())
                # save ref inside gen
                ref_gen = xt_full[0, :prefix_len]

                gen = xt_full[0, prefix_len:].detach().cpu()

                # Truncate at first EOS, then drop any remaining MASK (if any)
                gen = truncate_at_first_eos(gen, eos_id)
                gen = gen[gen != mask_id]

                # Prepare shape for codec decode (matches your training wrapper usage)
                gen_for_codec = gen.to(device).unsqueeze(0).unsqueeze(0)  # [1,1,T]

                gen_for_codec = gen_for_codec.detach()

                # Decode to waveform
                wav = codec.decode_code(gen_for_codec).detach()

                wav_ref = codec.decode_code(ref_gen.unsqueeze(0).unsqueeze(0)).detach()
                out_gen_ref = os.path.join(args.output_dir, f"gen_ref_{idx}-nsf{steps}.wav")
                torchaudio.save(out_gen_ref, wav_ref.squeeze(0).cpu(), saving_sr)

            # Save
            torchaudio.save(out_wav, wav.squeeze(0).cpu(), saving_sr)

        # except Exception as e:
        #     print(f"[row {idx}] error: {e}")
        #     continue


if __name__ == "__main__":
    main()
