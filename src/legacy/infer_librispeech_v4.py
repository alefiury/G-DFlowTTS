import os
import time
import math
import argparse
import warnings
from dataclasses import dataclass
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
    remask_noise: float = 0.0,      # adds stochastic remasking (optional)
    use_pfg: bool = False,
    gamma: float = 1.0,             # PFG strength in log-rate space; 1=conditional, 0=unconditional
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
    T = prefix_len + int(suffix_len)

    # init all MASK, then pin prefix
    xt = torch.full((1, T), mask_id, device=device, dtype=torch.long)
    xt[:, :prefix_len] = codes_ref_1d.unsqueeze(0)

    # add eos token at the end
    xt[0, -1] = eos_id

    # attention mask for audio (all valid here; we are not padding inside sampler)
    audio_att_mask = torch.ones_like(xt, dtype=torch.bool, device=device)

    dt = 1.0 / max(1, int(steps))
    eps = 1e-12

    def temp_at(t_lin: float) -> float:
        if temp_schedule == "dfm36":
            # Eq. (36) style schedule: T(t) = x1_temp * (1-t)^2
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

    for k in range(int(steps)):
        t_lin = k * dt
        t = torch.full((1,), float(t_lin), device=device, dtype=torch.float32)

        sched = path.scheduler(t)
        alpha_t = sched.alpha_t  # [1]
        dalpha_t = sched.d_alpha_t.clamp_min(1e-6)  # [1]

        # lambda(t) = d alpha / (1 - alpha)
        lam = (dalpha_t / (1.0 - alpha_t).clamp_min(1e-6))  # [1]
        lam_scalar = float(lam.item())

        # temperature
        Tsoft = temp_at(t_lin)

        # unconditional / conditional logits
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
            if pad_id is not None and not bool(getattr(config.datasets, "use_eos_as_pad", False)):
                logits_u[..., pad_id] = -torch.inf
            probs_u = torch.softmax(logits_u / Tsoft, dim=-1)
        else:
            probs_u = None

        # Avoid sampling MASK as a target token (should already be near-zero, but make it explicit)
        probs_c[..., mask_id] = 0.0
        probs_c = probs_c / probs_c.sum(dim=-1, keepdim=True).clamp_min(eps)
        if probs_u is not None:
            probs_u[..., mask_id] = 0.0
            probs_u = probs_u / probs_u.sum(dim=-1, keepdim=True).clamp_min(eps)

        # Build rates:
        # For mask-source, only masked positions unmask with rate = lam * p(token).
        xt_is_mask = (xt == mask_id).unsqueeze(-1).float()  # [1,T,1]

        base_r = (1.0 + float(remask_noise) * float(t_lin)) * lam_scalar  # scalar

        R_c = xt_is_mask * probs_c * base_r  # [1,T,S]

        if use_pfg:
            R_u = xt_is_mask * probs_u * base_r
        else:
            R_u = None

        # Optional remasking noise: allow non-mask -> mask transitions.
        # This is *not* part of the pure mixture path, but can add stochasticity.
        if remask_noise > 0.0:
            mask_one_hot = torch.zeros((S,), device=device, dtype=R_c.dtype)
            mask_one_hot[mask_id] = 1.0
            xt_not_mask = (1.0 - xt_is_mask)  # [1,T,1]
            R_c = R_c + xt_not_mask * mask_one_hot.view(1, 1, S) * float(remask_noise)
            if R_u is not None:
                R_u = R_u + xt_not_mask * mask_one_hot.view(1, 1, S) * float(remask_noise)

        # print("5"*100)

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

        # Jump probability for each position
        p_jump = 1.0 - torch.exp(-dt * hazard)  # [1,T]
        do_jump = (torch.rand_like(p_jump) < p_jump)  # [1,T] bool

        # Never jump on the pinned prefix
        do_jump[:, :prefix_len] = False

        if do_jump.any():
            # Build jump distributions only where hazard > 0 AND do_jump==True
            hazard_safe = hazard.clamp_min(1e-9)                     # [1,T]
            q = R_off / hazard_safe.unsqueeze(-1)                    # [1,T,S]

            q2 = q.view(-1, S)                                       # [T,S]
            jump_idx = do_jump.view(-1).nonzero(as_tuple=False).squeeze(-1)  # [Nj]

            q_jump = q2.index_select(0, jump_idx)                    # [Nj,S]
            # sanitize numerics
            q_jump = torch.nan_to_num(q_jump, nan=0.0, posinf=0.0, neginf=0.0)
            q_jump = q_jump.clamp_min(0.0)
            q_jump = q_jump / q_jump.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            sampled = torch.multinomial(q_jump, 1).squeeze(-1)       # [Nj]

            xt_flat = xt.view(-1)
            xt_flat[jump_idx] = sampled.to(xt_flat.dtype)
            xt = xt_flat.view_as(xt)


        # count number of masked positions in suffix
        num_masked = (xt[:, prefix_len:] == mask_id).sum().item()
        # print(f"Step {k+1}/{steps}, masked positions in suffix: {num_masked}")

        # Optional early stop if everything (suffix) is unmasked and no remasking
        if remask_noise <= 0.0:
            if (xt[:, prefix_len:] == mask_id).sum().item() == 0:
                # print(f"All positions unmasked at step {k+1}/{steps}, stopping early.")
                break

        xt[0, -1] = eos_id  # ensure EOS stays at the end
        xt[:, :prefix_len] = codes_ref_1d.unsqueeze(0)  # ensure prefix stays pinned

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


def load_codes(filepath: str) -> torch.Tensor:
    """
    Load pre-extracted codes from disk (if you saved them as tensors).
    Adjust this if you used a different saving format.
    """
    data = load_file(filepath)
    return data["fsq_codes"]


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

            # target_filepath = str(row["filepath"])
            # reference_filepath = str(row["reference"])
            codes_ref = load_codes(ref_filepath_codec).squeeze()
            if codes_ref.ndim != 1:
                codes_ref = codes_ref.reshape(-1)
            codes_ref = codes_ref.long().to(device)

            # Build text inputs using the SAME tokenizer as training
            text_ids, text_att_mask, _tok = build_text_inputs(config, text, text_ref, device)

            # Oracle length (suffix length)
            oracle_len = None
            if args.use_oracle_length:
                # oracle_codes = torch.load(filepath_codec).squeeze()
                # oracle_codes = extract_codes(codec, target_filepath).squeeze()
                oracle_codes = load_codes(filepath_codec).squeeze()
                if oracle_codes.ndim != 1:
                    oracle_codes = oracle_codes.reshape(-1)
                oracle_len = int(oracle_codes.numel())
                if args.oracle_add_eos:
                    oracle_len += 1

            # print(f"\n\n oracle_len: {oracle_len}-{oracle_len/50} \n\n")

            # Predict suffix length if needed
            if oracle_len is None:
                suffix_len = 2048
                # print(f"[row {idx}] using fixed suffix length: {suffix_len}")
            else:
                suffix_len = oracle_len

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
                    )
                    end_time = time.time()
                    total_pred_time = end_time - start_time

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

                total_wav_length = wav.shape[-1] / saving_sr
                rtf = total_pred_time / total_wav_length if total_wav_length > 0 else float("inf")

                rtf_tuples.append((idx, steps, total_pred_time, total_wav_length, rtf))
                # Save
                torchaudio.save(out_wav, wav.squeeze(0).cpu(), saving_sr)

        except Exception as e:
            print(f"[row {idx}] error: {e}")
            continue

    # Save RTFs
    rtf_df = pd.DataFrame(rtf_tuples, columns=["idx", "steps", "total_pred_time", "total_wav_length", "rtf"])
    rtf_df.to_csv(os.path.join(args.output_dir, "rtf_results.csv"), index=False)


if __name__ == "__main__":
    main()
