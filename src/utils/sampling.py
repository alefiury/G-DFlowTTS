"""Mask, Sample, Revise: CTMC sampler and inference helpers for G-DFlowTTS.

- CTMC tau-leaping sampler for the mask-source path (jump prob = 1 - exp(-dt * hazard))
- Optional predictor-free guidance (PFG), mixing conditional/unconditional rates in log space
- Optional Temporal Score Rescaling (TSR) applied to conditional and unconditional logits
- SC-ReMask: schedule-constrained CTMC remasking
    * sigma_max constraint from the schedule (alpha_t, alpha_s)
    * switch time (tswitch) + rescale (eta_rescale, eta_cap)
    * implemented as a token -> MASK CTMC rate, so it enters the hazard and jump decisions
    * optional confidence-based weighting (low-confidence tokens remask more)

Notes:
- With use_sc_remask, keep remask_noise = 0.0 (SC-ReMask replaces the ad-hoc remasking noise).
- Early stopping is disabled when remasking is enabled, since later remasks can still revise tokens.
"""

import math
import warnings
from typing import Optional, List, Tuple

warnings.filterwarnings("ignore")

import torch
from safetensors.torch import load_file
import torchaudio
from torchaudio import transforms as T

from transformers import AutoTokenizer

from utils.neucodec import NeuCodec
from modules.wrappers.dp_wrapper import DurationPredictorWrapper

# Flow-matching path/scheduler
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler

# Optional KO scheduler if you used it
try:
    from modules.gdflowtts.flow import KOConvexScheduler
except Exception:
    KOConvexScheduler = None

# Codecs
try:
    from xcodec2.modeling_xcodec2 import XCodec2Model
except Exception:
    XCodec2Model = None


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

    if config.datasets.type not in ["hf_text_tokenizer", "hf_streaming_text_tokenizer", "hf_streaming_codes"]:
        raise ValueError(f"Invalid dataset type: {config.datasets.type}")
    tok = AutoTokenizer.from_pretrained(config.datasets.text_tokenizer_name)
    tok = ensure_gpt2_padding(tok)
    enc = tok(augmented, return_tensors="pt")
    text_ids = enc["input_ids"].to(device)
    text_att_mask = enc["attention_mask"].bool().to(device)
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


def compute_sigma_sc_remask(
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
    SC-ReMask constraint:
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
# Correct CTMC sampler (mask-source) + optional PFG + TSR + SC-ReMask remasking
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
    # SC-ReMask
    use_sc_remask: bool = False,
    sc_remask_eta_rescale: float = 0.3,
    sc_remask_eta_cap: float = 0.5,
    sc_remask_tswitch: float = 0.7,
    sc_remask_use_conf: bool = False,
    sc_remask_conf_threshold: float = 0.35,
    sc_remask_beta: float = 2.0,
    sc_remask_strength: float = 1.0,
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

    # put EOS at final position (kept pinned)
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

        # also compute alpha at next time for SC-ReMask sigma_max constraint
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

        # Old ad-hoc remasking noise (token -> MASK)
        if remask_noise > 0.0:
            mask_one_hot = torch.zeros((S,), device=device, dtype=R_c.dtype)
            mask_one_hot[mask_id] = 1.0
            xt_not_mask = (1.0 - xt_is_mask)  # [1,T,1]
            R_c = R_c + xt_not_mask * mask_one_hot.view(1, 1, S) * float(remask_noise)
            if R_u is not None:
                R_u = R_u + xt_not_mask * mask_one_hot.view(1, 1, S) * float(remask_noise)

        # PFG mixing in log-rate space
        if use_pfg:
            logRc = torch.log(R_c + 1e-9)
            logRu = torch.log(R_u + 1e-9)
            R_mix = torch.exp(float(gamma) * logRc + (1.0 - float(gamma)) * logRu)
        else:
            R_mix = R_c

        # ----------------------------
        # SC-ReMask remasking (principled token -> MASK)
        # ----------------------------
        if use_sc_remask:
            sigma = compute_sigma_sc_remask(
                alpha_t=alpha_t,
                alpha_s=alpha_s,
                eta_rescale=float(sc_remask_eta_rescale),
                eta_cap=float(sc_remask_eta_cap),
                tswitch=float(sc_remask_tswitch),
                t_lin=float(t_lin),
            )  # [1]

            sigma_scalar = float(sigma.item())
            if sigma_scalar > 0.0:
                # Convert per-step remask probability sigma into a CTMC rate for this dt:
                #   p = 1 - exp(-dt * r) = sigma  =>  r = -log(1 - sigma) / dt
                r_base = -math.log(max(1e-9, 1.0 - sigma_scalar)) / max(1e-12, dt)

                # eligible positions: suffix, non-mask, non-EOS, not the final EOS
                eligible = torch.ones_like(xt, dtype=torch.bool, device=device)
                eligible[:, :prefix_len] = False
                eligible[:, -1] = False
                eligible = eligible & (xt != mask_id) & (xt != eos_id)

                # Optional confidence weighting (remask low-confidence tokens more)
                if sc_remask_use_conf:
                    if use_pfg:
                        # geometric mixture consistent with log-rate PFG
                        logp_c = torch.log(probs_c + eps)
                        logp_u = torch.log(probs_u + eps)
                        logp_mix = float(gamma) * logp_c + (1.0 - float(gamma)) * logp_u
                        probs_mix = torch.exp(logp_mix)
                        probs_mix = probs_mix / probs_mix.sum(dim=-1, keepdim=True).clamp_min(eps)
                    else:
                        probs_mix = probs_c

                    p_cur = probs_mix.gather(-1, xt[..., None].clamp(0, S - 1)).squeeze(-1)  # [1,T]
                    thr = float(sc_remask_conf_threshold)
                    beta = float(sc_remask_beta)

                    # weight in [0,1] where 1 = very low confidence, 0 = confident
                    # if p_cur >= thr => 0
                    low = (p_cur < thr).float()
                    w = low * ((thr - p_cur) / max(1e-9, thr)).clamp(0.0, 1.0) ** beta
                    w = (w * float(sc_remask_strength)).clamp(0.0, 1.0)
                else:
                    w = torch.ones_like(xt, dtype=torch.float32, device=device)

                # Add token->MASK rate on eligible positions
                # r_pos = r_base * w
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

        # Never jump on prefix and final EOS
        do_jump[:, :prefix_len] = False
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

        # Keep EOS at the end and prefix pinned
        xt[0, -1] = eos_id
        xt[:, :prefix_len] = codes_ref_1d.unsqueeze(0)

        # Early stop only if NO remasking can happen
        # (SC-ReMask could still remask later and fix errors, especially with tswitch)
        if (remask_noise <= 0.0) and (not use_sc_remask):
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
