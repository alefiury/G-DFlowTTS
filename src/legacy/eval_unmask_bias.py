#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import math
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import AutoTokenizer

# Your modules (same as your inference script)
from modules.pl_wrapper import DFMTTSWrapper
from utils.tokenizer import VoiceBpeTokenizer
from tqdm import tqdm
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler

try:
    from modules.flow import KOConvexScheduler
except Exception:
    KOConvexScheduler = None


# ----------------------------
# Utils
# ----------------------------

def seed_everything(seed: int = 0) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
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
        return text_ids, text_att_mask
    else:
        tok = VoiceBpeTokenizer(vocab_file=config.datasets.vocab_file)
        text_ids = torch.tensor(tok.encode(augmented, lang="en-us"), device=device).unsqueeze(0)
        text_att_mask = torch.ones_like(text_ids, dtype=torch.bool, device=device)
        return text_ids, text_att_mask

def build_path_from_config(config):
    sched_type = str(getattr(config, "scheduler_type", "polynomial")).lower()
    if sched_type == "ko":
        if KOConvexScheduler is None:
            raise RuntimeError("KOConvexScheduler not importable, but scheduler_type=ko.")
        return MixtureDiscreteProbPath(scheduler=KOConvexScheduler())
    return MixtureDiscreteProbPath(scheduler=PolynomialConvexScheduler(n=1.0))

def total_vocab_size(config) -> int:
    return int(config.datasets.audio_vocab_size) + int(config.model.audio_add_token)

def load_codes_1d(path: str) -> torch.Tensor:
    x = torch.load(path, map_location="cpu")
    if isinstance(x, dict) and "codes" in x:
        x = x["codes"]
    x = x.squeeze()
    if x.ndim != 1:
        x = x.reshape(-1)
    return x.long()

def rankdata_average_ties(a: np.ndarray) -> np.ndarray:
    """
    Like scipy.stats.rankdata(method="average"), but minimal and fast.
    Returns ranks in [0..n-1] with average ranks for ties.
    """
    n = a.size
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[order[j + 1]] == a[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j)
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    return ranks

def spearmanr_fast(x: np.ndarray, y: np.ndarray) -> float:
    rx = rankdata_average_ties(x.astype(np.float64))
    ry = rankdata_average_ties(y.astype(np.float64))
    vx = rx - rx.mean()
    vy = ry - ry.mean()
    denom = (np.sqrt((vx * vx).sum()) * np.sqrt((vy * vy).sum()))
    if denom <= 1e-12:
        return float("nan")
    return float((vx * vy).sum() / denom)

def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    # least squares slope
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    vx = x - x.mean()
    denom = (vx * vx).sum()
    if denom <= 1e-12:
        return float("nan")
    return float((vx * (y - y.mean())).sum() / denom)

def pairwise_precedence_approx(times: np.ndarray, num_pairs: int, rng: np.random.Generator) -> float:
    """
    Approx fraction of random pairs (i<j) where t[i] < t[j].
    0.5 ~ no ordering bias; >0.5 indicates left->right ordering.
    """
    L = times.size
    if L < 2:
        return float("nan")
    i = rng.integers(0, L, size=num_pairs, endpoint=False)
    j = rng.integers(0, L, size=num_pairs, endpoint=False)
    ii = np.minimum(i, j)
    jj = np.maximum(i, j)
    mask = (ii != jj)
    if mask.sum() == 0:
        return float("nan")
    ii = ii[mask]
    jj = jj[mask]
    return float((times[ii] < times[jj]).mean())

def adjacent_monotonicity(times: np.ndarray) -> float:
    if times.size < 2:
        return float("nan")
    return float((times[:-1] <= times[1:]).mean())


# ----------------------------
# Sampler with trace (absorbing MASK) + normalized CFG/PFG (log-prob space)
# ----------------------------

@torch.inference_mode()
def sample_mask_ctmc_with_trace(
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
    temp_schedule: str = "dfm36",      # "dfm36" or "constant"
    remask_noise: float = 0.0,
    # normalized CFG/PFG
    use_pfg: bool = False,
    gamma: float = 1.0,               # 0=uncond, 1=cond, >1 stronger
    # control
    pin_eos: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Returns:
      xt: [1, T] final tokens
      first_unmask_step: [T] int64, -1 for positions that never unmasked (or were never MASK)
      steps_taken: actual number of steps executed (<= steps if early stop)
    """
    S = total_vocab_size(config)
    mask_id = int(config.datasets.audio_mask_token)
    eos_id = int(getattr(config.datasets, "audio_eos_token", -1))

    prefix_len = int(codes_ref_1d.numel())
    T = prefix_len + int(suffix_len)

    xt = torch.full((1, T), mask_id, device=device, dtype=torch.long)
    xt[:, :prefix_len] = codes_ref_1d.unsqueeze(0)

    if pin_eos and eos_id >= 0:
        xt[0, -1] = eos_id

    audio_att_mask = torch.ones_like(xt, dtype=torch.bool, device=device)

    dt = 1.0 / max(1, int(steps))
    eps = 1e-12

    # first_unmask_step[i] = first k where token i went MASK -> non-MASK
    first_unmask_step = torch.full((T,), -1, device=device, dtype=torch.long)

    def temp_at(t_lin: float) -> float:
        if temp_schedule == "dfm36":
            return max(1e-3, float(x1_temp) * (1.0 - float(t_lin)) ** 2)
        return max(1e-3, float(x1_temp))

    if use_pfg:
        cond_drop_prob = float(getattr(config.datasets, "cond_drop_prob", 0.0))
        if cond_drop_prob <= 0.0:
            raise RuntimeError(
                "use_pfg=True but config.datasets.cond_drop_prob==0.0; "
                "checkpoint likely lacks unconditional branch."
            )

    steps_taken = 0
    for k in range(int(steps)):
        steps_taken = k + 1
        t_lin = k * dt
        t = torch.full((1,), float(t_lin), device=device, dtype=torch.float32)

        sched = path.scheduler(t)
        alpha_t = sched.alpha_t
        dalpha_t = sched.d_alpha_t.clamp_min(1e-6)

        lam = (dalpha_t / (1.0 - alpha_t).clamp_min(1e-6))
        lam_scalar = float(lam.item())

        Tsoft = temp_at(t_lin)

        prev_xt = xt.clone()

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

            # normalized CFG/PFG in log-prob space
            logp_c = F.log_softmax(logits_c / Tsoft, dim=-1)
            logp_u = F.log_softmax(logits_u / Tsoft, dim=-1)
            logp_guided = logp_u + float(gamma) * (logp_c - logp_u)
            probs = torch.softmax(logp_guided, dim=-1)
        else:
            probs = torch.softmax(logits_c / Tsoft, dim=-1)

        # forbid MASK as a target
        probs[..., mask_id] = 0.0
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)

        # absorbing (mask-source): only masked positions can jump to tokens
        xt_is_mask = (xt == mask_id).unsqueeze(-1).float()  # [1,T,1]
        base_r = (1.0 + float(remask_noise) * float(t_lin)) * lam_scalar
        R = xt_is_mask * probs * base_r  # [1,T,S]

        # optional remasking (if you want to test)
        if remask_noise > 0.0:
            mask_one_hot = torch.zeros((S,), device=device, dtype=R.dtype)
            mask_one_hot[mask_id] = 1.0
            xt_not_mask = (1.0 - xt_is_mask)  # [1,T,1]
            R = R + xt_not_mask * mask_one_hot.view(1, 1, S) * float(remask_noise)

        # remove diagonal
        R_off = R.clone()
        R_off.scatter_(-1, xt[..., None], 0.0)

        hazard = R_off.sum(dim=-1)  # [1,T]
        p_jump = 1.0 - torch.exp(-dt * hazard)
        do_jump = (torch.rand_like(p_jump) < p_jump)

        # pin prefix (and EOS if desired)
        do_jump[:, :prefix_len] = False
        if pin_eos and eos_id >= 0:
            do_jump[:, -1] = False

        if do_jump.any():
            hazard_safe = hazard.clamp_min(1e-9)
            q = R_off / hazard_safe.unsqueeze(-1)

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

        # record first MASK -> non-MASK events
        newly_unmasked = (prev_xt[0] == mask_id) & (xt[0] != mask_id)
        to_set = newly_unmasked & (first_unmask_step == -1)
        first_unmask_step[to_set] = k  # 0-based step index

        # early stop if all suffix unmasked and no remask
        if remask_noise <= 0.0:
            if (xt[:, prefix_len:] == mask_id).sum().item() == 0:
                break

    return xt, first_unmask_step, steps_taken


# ----------------------------
# Main evaluation
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--metadata_csv", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--gpu", type=int, default=0)

    ap.add_argument("--nsf", type=str, default="256", help="Comma-separated steps, e.g. '128,256'")
    ap.add_argument("--max_rows", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)

    # lengths
    ap.add_argument("--use_oracle_length", action="store_true")
    ap.add_argument("--oracle_add_eos", action="store_true")
    ap.add_argument("--fixed_suffix_len", type=int, default=1024)

    # sampling knobs
    ap.add_argument("--x1_temp", type=float, default=1.0)
    ap.add_argument("--temp_schedule", type=str, default="dfm36", choices=["dfm36", "constant"])
    ap.add_argument("--remask_noise", type=float, default=0.0)

    # guidance knobs (normalized CFG/PFG in log-prob space)
    ap.add_argument("--use_pfg", action="store_true")
    ap.add_argument("--gamma", type=float, default=1.0)

    # evaluation options
    ap.add_argument("--exclude_last_eos", action="store_true", default=True)
    ap.add_argument("--num_pair_samples", type=int, default=20000)

    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)

    config = OmegaConf.load(args.config)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load model
    model = DFMTTSWrapper.load_from_checkpoint(
        args.checkpoint,
        config=config,
        map_location=device,
        strict=False,
        weights_only=False,
    ).to(device)
    model.eval()

    path = build_path_from_config(config)

    df = pd.read_csv(args.metadata_csv)
    if args.max_rows > 0:
        df = df.iloc[: args.max_rows].copy()

    # Required columns
    required = ["text", "ref_text", "filepath_codec", "reference_codec"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"metadata_csv missing column: {c}")

    steps_list = parse_int_list(args.nsf)

    # Aggregate containers per steps
    all_rows = []

    for steps in steps_list:
        sum_curve = None
        cnt_curve = None

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            text = str(row["text"])
            text_ref = None if pd.isna(row["ref_text"]) else str(row["ref_text"])

            ref_codec_path = str(row["reference_codec"])
            tgt_codec_path = str(row["filepath_codec"])

            codes_ref = load_codes_1d(ref_codec_path).to(device)

            text_ids, text_att_mask = build_text_inputs(config, text, text_ref, device)

            if args.use_oracle_length:
                oracle = load_codes_1d(tgt_codec_path)
                oracle_len = int(oracle.numel())
                if args.oracle_add_eos:
                    oracle_len += 1
                suffix_len = oracle_len
            else:
                suffix_len = int(args.fixed_suffix_len)

            xt, first_unmask_step, steps_taken = sample_mask_ctmc_with_trace(
                config=config,
                model=model,
                path=path,
                text_ids=text_ids,
                text_att_mask=text_att_mask,
                codes_ref_1d=codes_ref,
                suffix_len=suffix_len,
                steps=int(steps),
                device=device,
                x1_temp=float(args.x1_temp),
                temp_schedule=str(args.temp_schedule),
                remask_noise=float(args.remask_noise),
                use_pfg=bool(args.use_pfg),
                gamma=float(args.gamma),
                pin_eos=True,
            )

            prefix_len = int(codes_ref.numel())
            # suffix positions
            t_steps = first_unmask_step[prefix_len:].detach().cpu().numpy().astype(np.int32)

            # if eos pinned at last position, it's non-mask from step 0; drop it
            if args.exclude_last_eos and t_steps.size > 0:
                t_steps = t_steps[:-1]

            # handle never-unmasked (shouldn't happen if it fully unmasked); set to steps_taken
            t_steps = np.where(t_steps < 0, steps_taken, t_steps)

            pos = np.arange(t_steps.size, dtype=np.int32)

            sp = spearmanr_fast(pos, t_steps) if t_steps.size >= 2 else float("nan")
            sl = linear_slope(pos, t_steps) if t_steps.size >= 2 else float("nan")

            # left-right gap: last quartile mean - first quartile mean
            if t_steps.size >= 8:
                q = t_steps.size // 4
                left_mean = float(t_steps[:q].mean())
                right_mean = float(t_steps[-q:].mean())
                lr_gap = right_mean - left_mean
            else:
                lr_gap = float("nan")

            adj_mono = adjacent_monotonicity(t_steps)
            pair_prec = pairwise_precedence_approx(t_steps, args.num_pair_samples, rng)

            all_rows.append({
                "steps": int(steps),
                "row_idx": int(idx),
                "suffix_len_eval": int(t_steps.size),
                "steps_taken": int(steps_taken),
                "spearman_pos_vs_unmask": sp,
                "slope_pos_vs_unmask": sl,
                "left_right_gap": lr_gap,
                "adjacent_monotonicity": adj_mono,
                "pairwise_precedence": pair_prec,
            })

            # aggregate curve
            L = t_steps.size
            if sum_curve is None:
                sum_curve = np.zeros((L,), dtype=np.float64)
                cnt_curve = np.zeros((L,), dtype=np.int64)
            if L > sum_curve.size:
                new_sum = np.zeros((L,), dtype=np.float64)
                new_cnt = np.zeros((L,), dtype=np.int64)
                new_sum[:sum_curve.size] = sum_curve
                new_cnt[:cnt_curve.size] = cnt_curve
                sum_curve, cnt_curve = new_sum, new_cnt

            sum_curve[:L] += t_steps.astype(np.float64)
            cnt_curve[:L] += 1

        # save mean curve for this steps
        mean_curve = sum_curve / np.maximum(cnt_curve, 1)
        np.save(os.path.join(args.output_dir, f"mean_unmask_curve_nsf{steps}.npy"), mean_curve)

    # Save per-sample metrics
    out_csv = os.path.join(args.output_dir, "unmask_bias_metrics.csv")
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)

    # Print summary per steps
    metrics = pd.DataFrame(all_rows)
    print("\n=== Summary by steps ===")
    for steps in steps_list:
        m = metrics[metrics["steps"] == int(steps)]
        print(
            f"\n[nsf={steps}] N={len(m)} | "
            f"mean_spearman={m['spearman_pos_vs_unmask'].mean():.4f} | "
            f"mean_slope={m['slope_pos_vs_unmask'].mean():.4f} | "
            f"mean_lr_gap={m['left_right_gap'].mean():.4f} | "
            f"mean_adj_mono={m['adjacent_monotonicity'].mean():.4f} | "
            f"mean_pair_prec={m['pairwise_precedence'].mean():.4f}"
        )

    print(f"\nWrote: {out_csv}")
    print("Wrote: mean_unmask_curve_nsf*.npy")


if __name__ == "__main__":
    main()
