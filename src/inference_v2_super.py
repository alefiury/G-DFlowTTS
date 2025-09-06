import os
import json
import logging
import argparse
import warnings
from pprint import pprint
from typing import Tuple, Optional

warnings.filterwarnings("ignore")

import wandb
import torch
import pandas as pd
from torch import nn, Tensor
import torch.nn.functional as F
from tqdm import tqdm
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


# -------------------------
# Utilities / helpers
# -------------------------
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
) -> int:
    """
    Predict remaining duration given a BOS+codes_ref prefix.
    Returns an integer length for the to-be-generated continuation.
    """
    bos_vec = codes_ref.new_full((1,), 65536)
    codes_ref = torch.cat((bos_vec, codes_ref), dim=0)   # [C, dur+1]

    remaining_duration = duration_model(
        text_ids=text_ids,
        audio_ids=codes_ref.unsqueeze(0).to(device)
    )
    return torch.argmax(remaining_duration[:, -1], dim=-1).item()


# -------------------------
# Core DFM inference (CTMC Euler + schedulers + corrector)
# -------------------------
@torch.inference_mode()
def inference(
    config,
    model,
    duration_model,
    tokenizer,
    sentence: str,
    nsf: int = 256,
    text_ref: Optional[str] = None,
    codes_ref: Optional[Tensor] = None,
    sequence_length: int = 300,
    device: torch.device = torch.device("cuda"),
    # ---- DFM path schedulers ----
    # options: "linear" | "cosine" | "smoothstep" | "cubic" | "cubic_poly"
    scheduler: str = "cubic_poly",
    # Paper Eq. (cubic polynomial path with end-derivative controls a,b):
    # κ(t) = -2t^3 + 3t^2 + a(t^3 - 2t^2 + t) + b(t^3 - t^2)
    scheduler_a: float = 0.0,  # paper recommended for text: a=0
    scheduler_b: float = 2.0,  # paper recommended for text: b=2 -> κ(t)=t^2
    # avoid t=1 singularities
    time_epsilon: float = 1e-6,
    # ---- Corrector (re-mask) ----
    corrector_steps: int = 0,               # 0 disables corrector
    corrector_remask_beta: float = 0.0,     # base re-mask strength; 0 disables
    # α_t = 1 + α * t^a * (1-t)^b (time scaling for corrector)
    corrector_alpha: float = 0.0,           # set >0 (e.g., 10–20) to enable scaling
    corrector_a: float = 0.0,
    corrector_b: float = 0.0,
    # ---- Token bans during sampling ----
    ban_pad_mask_during_sampling: bool = True,
    ban_eos_during_sampling: bool = False,
    extra_banned_tokens: Tuple[int, ...] = (),
):
    """
    Discrete Flow Matching (masked-source) sampler using a CTMC Euler step (hazard 1-exp(-dt*λ)).
    Keeps codes_ref as a hard prefix and generates the continuation.

    Returns:
        Tensor of shape [1, total_len] with token ids.
    """

    # Build text condition
    augmented_sentence = f"{text_ref} {sentence}" if text_ref is not None else sentence
    text_ids = tokenizer.encode(augmented_sentence, lang="en-us")
    # text_ids = tokenizer.encode(augmented_sentence, lang="pt-br")
    text_ids = torch.tensor(text_ids).unsqueeze(0).to(device)

    # Vocab helpers
    mask_token = int(config.datasets.audio_mask_token)
    eos_token = int(config.datasets.audio_eos_token)
    pad_token = int(config.datasets.audio_pad_token)
    vocab_size = int(config.datasets.audio_vocab_size + config.model.add_token)

    # Path scheduler κ(t), κ̇(t)
    def kappa_and_dot(t: Tensor):
        # t in [0,1)
        if scheduler == "linear":
            k = t
            dk = torch.ones_like(t)
        elif scheduler == "cosine":
            # k = (1 - cos(pi t))/2, dk = (pi/2) sin(pi t)
            k = 0.5 * (1.0 - torch.cos(torch.pi * t))
            dk = 0.5 * torch.pi * torch.sin(torch.pi * t)
        elif scheduler == "smoothstep":
            # k = 3t^2 - 2t^3, dk = 6t - 6t^2
            k = t * t * (3 - 2 * t)
            dk = 6 * t * (1 - t)
        elif scheduler == "cubic":
            # ease-in: k = t^3, dk = 3 t^2
            k = t * t * t
            dk = 3 * t * t
        elif scheduler == "cubic_poly":
            # Paper cubic polynomial family with parameters a,b
            # κ(t) = -2t^3 + 3t^2 + a(t^3 - 2t^2 + t) + b(t^3 - t^2)
            a = float(scheduler_a)
            b = float(scheduler_b)
            k = (-2 * t**3 + 3 * t**2
                 + a * (t**3 - 2 * t**2 + t)
                 + b * (t**3 - t**2))
            dk = (-6 * t**2 + 6 * t
                  + a * (3 * t**2 - 4 * t + 1)
                  + b * (3 * t**2 - 2 * t))
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}")

        # Clamp away from {0,1} to avoid division by zero in rates
        eps = 1e-8
        k = k.clamp(eps, 1 - eps)
        dk = dk.clamp(min=eps)
        return k, dk

    # Source distribution (all-mask initialization)
    source_distribution = MaskedSourceDistribution(mask_token=mask_token)

    # Predict target continuation length from duration model (keeps your existing logic)
    if codes_ref is None:
        raise ValueError("codes_ref must be provided for prefix conditioning.")
    sequence_length = get_remaining_duration(
        duration_model,
        text_ids=text_ids,
        codes_ref=codes_ref,
        device=device
    )

    print("-" * 100)
    print("Predicted sequence length:", sequence_length)
    print("codes_ref.size(0):", codes_ref.size(0))

    # Initialize x_t with masks, paste reference prefix
    orig_ref_code_len = codes_ref.size(0)
    xt = source_distribution.sample(
        (1, sequence_length + orig_ref_code_len), device=device
    )
    xt[..., :orig_ref_code_len] = codes_ref[..., :orig_ref_code_len]

    # Time grid [0, 1 - eps]
    num_steps = int(nsf)
    dt = 1.0 / max(num_steps, 1)
    t_grid = torch.linspace(0.0, 1.0 - time_epsilon, num_steps, device=device)

    # Prefix mask (never change)
    L = xt.shape[-1]
    idx = torch.arange(L, device=device)
    is_prefix = idx < orig_ref_code_len
    not_prefix = ~is_prefix

    # Build banned token list
    banned_list = list(extra_banned_tokens)
    if ban_pad_mask_during_sampling:
        banned_list.extend([mask_token, pad_token])
    if ban_eos_during_sampling:
        banned_list.append(eos_token)
    banned_tokens = torch.tensor(sorted(set(int(t) for t in banned_list)),
                                 device=device, dtype=torch.long) if len(banned_list) else None

    # Text attention mask (False means "not padded")
    text_att_mask = text_ids.new_zeros((1, text_ids.size(1)), dtype=torch.bool)

    for step, t_val in enumerate(t_grid):
        t_tensor = xt.new_full((1,), float(t_val), dtype=torch.float32)  # shape [1]
        k, dk = kappa_and_dot(t_tensor)             # tensors of shape [1]
        k = float(k.item())
        dk = float(dk.item())

        # Predictor (forward): probability denoiser p1|t
        logits = model(
            x_t=xt,
            text_ids=text_ids,
            text_att_mask=text_att_mask,
            time=t_tensor,
            drop_text=False
        )
        # precision matters before softmax
        p1 = torch.softmax(logits.float(), dim=-1)  # [1, L, V]
        cur = xt.squeeze(0)                         # [L]

        # CTMC rate: λ^i = (kdot / (1-k)) * (1 - p1_i[z])
        ar = torch.arange(L, device=device)
        p_stay = p1[0, ar, cur]                     # [L]
        lam = (dk / (1.0 - k)) * (1.0 - p_stay)     # [L]

        # Change probability with hazard: 1 - exp(-dt * λ)
        p_change = 1.0 - torch.exp(-dt * lam)       # [L]
        # Never change prefix
        p_change = torch.where(is_prefix, torch.zeros_like(p_change), p_change)

        # Sample change mask
        change = (torch.rand_like(p_change) < p_change) & not_prefix

        if change.any():
            # Proposal: off-diagonal probs ∝ p1 with current token prob set to 0
            prop = p1[0].clone()                    # [L, V]
            prop[ar, cur] = 0.0

            # Optionally ban tokens (set prob to 0)
            if banned_tokens is not None and banned_tokens.numel() > 0:
                prop[..., banned_tokens] = 0.0

            # Normalize rows; detect degenerate rows
            prop_sum = prop.sum(-1, keepdim=True)   # [L,1]
            can_norm = (prop_sum.squeeze(-1) > 1e-12)

            # Only positions that request change and have nonzero mass can change
            can_change = change & can_norm

            if can_change.any():
                rows = torch.nonzero(can_change).squeeze(-1)  # [N]
                # Sample new tokens for those rows
                sampled = torch.multinomial(prop[rows], num_samples=1).squeeze(-1)  # [N]
                cur[rows] = sampled

        xt = cur.unsqueeze(0).long()

        # Corrector: optional re-mask (backward-like nudging to source)
        if corrector_steps > 0 and corrector_remask_beta > 0.0:
            for _ in range(corrector_steps):
                cur = xt.squeeze(0)
                need = not_prefix & (cur != mask_token)
                if need.any():
                    # Base backward rate ~ (dk / k)
                    rate_back = (dk / k) * torch.ones_like(cur, dtype=torch.float, device=device)
                    # α_t = 1 + α * t^a * (1-t)^b (time-shaped corrector)
                    alpha_t_scale = 1.0 + float(corrector_alpha) * \
                        (float(t_val) ** float(corrector_a)) * ((1.0 - float(t_val)) ** float(corrector_b))
                    # Small step (0.1*dt) to keep corrector gentle
                    p_to_mask = 1.0 - torch.exp(- (dt * 0.1) * (corrector_remask_beta * alpha_t_scale) * rate_back)
                    p_to_mask = p_to_mask.clamp(0.0, 0.999)
                    p_to_mask = torch.where(need, p_to_mask, torch.zeros_like(p_to_mask))
                    flip = (torch.rand_like(p_to_mask) < p_to_mask) & need
                    cur = torch.where(flip, torch.tensor(mask_token, device=device), cur)
                    xt = cur.unsqueeze(0).long()

        # Keep prefix hard-frozen
        xt[..., :orig_ref_code_len] = codes_ref[..., :orig_ref_code_len]

    return xt


# -------------------------
# Driver
# -------------------------
@torch.no_grad()
def main() -> None:
    output_dir = "agora-Vai"
    gpu = 0
    config_path = "/raid/aluno_alef/DFM-TTS-2/config/offline-bpe-text_cfg-en.yaml"
    pretrained_checkpoint = "/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/m1ejk3am/checkpoints/epoch=29-step=500000-val/loss_epoch=3.366.ckpt"

    duration_pred_config_path = "/raid/aluno_alef/DFM-TTS-2/config/duration_predictor_bpe_en.yaml"
    duration_pred_pretrained_checkpoint = "/raid/aluno_alef/DFM-TTS-2/src/Duration-Predictor-DFM-TTS/61g87haf/checkpoints/epoch=10-step=138116-val/loss_epoch=4.717.ckpt"

    config = OmegaConf.load(config_path)
    duration_pred_config = OmegaConf.load(duration_pred_config_path)

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    tokenizer = VoiceBpeTokenizer(vocab_file=config.datasets.vocab_file)
    model = DFMTTSWrapper.load_from_checkpoint(
        pretrained_checkpoint, config=config, map_location=device, strict=False
    ).to(device)
    model.eval()

    duration_model = DurationPredictorWrapper.load_from_checkpoint(
        duration_pred_pretrained_checkpoint,
        config=duration_pred_config,
        map_location=device,
        strict=False
    ).to(device)
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

    nsf_list = [128, 256, 512, 1024, 2048]

    os.makedirs(output_dir, exist_ok=True)

    for idx, sentence in enumerate(sentences):
        print(f"Processing sentence {idx + 1}/{len(sentences)}")
        for n in tqdm(nsf_list):
            text_ref = (
                "He spoke with an extreme Oxford accent, and when he was talking well, "
                "his face sometimes wore the rapt expression of a very emotional man listening to music."
            )
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
                # ---- DFM knobs ----
                scheduler="cubic_poly",   # paper path family
                scheduler_a=0.0,
                scheduler_b=2.0,          # => κ(t) = t^2 (paper's recommended for text)
                time_epsilon=1e-6,
                corrector_steps=1,
                corrector_remask_beta=0.05,
                corrector_alpha=12.0,     # try 0.0 first; 10–20 is a decent sweep range
                corrector_a=0.25,
                corrector_b=0.25,
                # token bans
                ban_pad_mask_during_sampling=False,
                ban_eos_during_sampling=False,
                extra_banned_tokens=(),
            )

            print(x_t)

            # Post-process: remove prefix and stop at first EOS (if any)
            x_t = x_t.squeeze(0)                     # [total_len]
            x_t = x_t[codes_ref.size(0):]            # drop reference prefix
            print("Generated (with prefix removed) shape:", x_t.shape)
            print("/" * 100)

            # Count EOS and find first index safely
            eos_mask = (x_t == config.datasets.audio_eos_token)
            eos_count = int(eos_mask.sum().item())
            eos_indices = torch.nonzero(eos_mask, as_tuple=False).squeeze(-1)

            if eos_indices.numel() > 0:
                first_eos = int(eos_indices[0].item())
                x_t = x_t[..., :first_eos]           # cut at first EOS

            print(f"Number of EOS tokens: {eos_count}, first EOS index: "
                  f"{None if eos_indices.numel()==0 else int(eos_indices[0].item())}")

            # Remove EOS / MASK / PAD that might remain
            x_t = x_t[x_t != config.datasets.audio_eos_token]
            x_t = x_t[x_t != config.datasets.audio_mask_token]
            x_t = x_t[x_t != config.datasets.audio_pad_token]

            # Decode to audio
            x_t = x_t.unsqueeze(0).unsqueeze(0)      # [1, 1, T]
            generated_audio = audio_codec.decode_code(x_t)
            out_path = f"{output_dir}/audio_{idx}-{n}.wav"
            torchaudio.save(out_path, generated_audio.squeeze(0).cpu(), 16000)
            print("Saved:", out_path)


if __name__ == "__main__":
    main()
