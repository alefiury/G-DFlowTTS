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
    bos_vec = codes_ref.new_full((1,), 65536)
    codes_ref = torch.cat((bos_vec, codes_ref), dim=0)   # [C, dur+1]

    remaining_duration = duration_model(
        text_ids=text_ids,
        audio_ids=codes_ref.unsqueeze(0).to(device)
    )
    return torch.argmax(remaining_duration[:, -1], dim=-1).item()


# def apply_vlg_ops(
#     x: torch.Tensor,                  # [B, L] tokens predicted at current step
#     mask_token: int,
#     expand_token: int,
#     delete_token: int,
#     max_len: int
# ):
#     """
#     Apply DreamOn heuristic in-place:
#     - <EXPAND> -> two <MASK> at same position
#     - <DELETE> -> remove that token
#     Returns a possibly length-changed, padded back to max_len with PAD
#     """
#     B, L = x.shape
#     out = []
#     for b in range(B):
#         seq = x[b].tolist()
#         new_seq = []
#         for t in seq:
#             if t == expand_token:
#                 new_seq.append(mask_token)
#                 new_seq.append(mask_token)
#             elif t == delete_token:
#                 # skip it (deletion)
#                 continue
#             else:
#                 new_seq.append(t)
#         new_seq = new_seq[:max_len]
#         out.append(torch.tensor(new_seq, device=x.device, dtype=x.dtype))
#     # pad to common length
#     maxL = min(max_len, max(s.numel() for s in out) if out else L)
#     padded = x.new_full((B, maxL), fill_value=mask_token)  # keep masked tail
#     for b, s in enumerate(out):
#         padded[b, :min(maxL, s.numel())] = s[:maxL]
#     return padded


def apply_vlg_ops(
    x: torch.Tensor,                  # [B, L]
    mask_token: int,
    expand_token: int,
    delete_token: int,
    max_len: int,
    edit_start: int = 0,              # first editable index (e.g., codes_ref_size)
    edit_end: Optional[int] = None,   # last editable index (exclusive); None => full length
):
    """
    Variable-length growth (expand) & shrink (delete), applied ONLY in [edit_start, edit_end).
    - <EXPAND>  -> replace with [MASK, MASK]
    - <DELETE>  -> remove the nearest real LEFT neighbor *and* the <DELETE> itself (no mask appended)
    Then re-pad with MASK to max_len (so new slots are fillable on later steps).
    """
    B, L = x.shape
    SENTINELS = {mask_token, expand_token, delete_token}

    out = []
    for b in range(B):
        seq = x[b].tolist()
        if edit_end is None or edit_end > len(seq):
            e_end = len(seq)
        else:
            e_end = edit_end

        new_seq = []

        # Copy prefix (non-editable head)
        if edit_start > 0:
            new_seq.extend(seq[:edit_start])

        # Work on editable window
        i = edit_start
        while i < e_end:
            t = seq[i]

            # EXPAND: replace with two MASKs
            if t == expand_token:
                new_seq.append(mask_token)
                new_seq.append(mask_token)
                i += 1
                continue

            # DELETE: drop left real neighbor + the DELETE itself
            if t == delete_token:
                # # find a real (non-sentinel) left neighbor inside the editable window *or* in prefix
                # j = len(new_seq) - 1
                # while j >= 0 and new_seq[j] in SENTINELS:
                #     j -= 1
                # if j >= 0:
                #     new_seq.pop(j)   # remove the real token
                # # skip the DELETE itself by not appending it
                i += 1
                continue

            # Otherwise keep the token
            new_seq.append(t)
            i += 1

        # Copy tail (non-editable)
        if e_end < len(seq):
            new_seq.extend(seq[e_end:])

        # truncate then pad with MASK so new slots are fillable next steps
        new_seq = new_seq[:max_len]
        padded = [mask_token] * max_len
        upto = min(len(new_seq), max_len)
        padded[:upto] = new_seq[:upto]
        out.append(torch.tensor(padded, device=x.device, dtype=x.dtype))

    return torch.stack(out, dim=0)



@torch.inference_mode()
def inference(
    config,
    model,
    tokenizer,
    sentence,
    nsf: int = 10,
    text_ref: str = None,
    codes_ref: Tensor = None,
    sequence_length: int = 300,
    device: torch.device = torch.device("cuda")
) -> Tensor:
    augmented_sentence = text_ref + " " + sentence
    # text_ids = tokenizer.encode(augmented_sentence, lang="en-us")
    text_ids = tokenizer.encode(augmented_sentence, lang="pt-br")

    text_ids = torch.tensor(text_ids).unsqueeze(0).to(device)
    max_length = config.datasets.max_audio_length
    vocab_size = config.datasets.audio_vocab_size + config.model.add_token

    source_distribution = MaskedSourceDistribution(
        mask_token=config.datasets.audio_mask_token
    )
    # Set the number of predictor steps (you can adjust this or read it from config)
    num_steps = nsf  # for example, 10 steps from t=0 to t=1
    num_corrector_steps = 10  # number of corrector iterations per predictor step

    # Create a time grid from 0 to 1 with (num_steps + 1) points
    t_init = 0.0
    t_final = 1.0
    time_grid = torch.linspace(t_init, t_final, num_steps + 1, device=device)

    # Initialize x_t; for example, using the masked source
    xt = source_distribution.sample((1, sequence_length + codes_ref.size(0)), device=device)
    orig_ref_code_len = codes_ref.size(0)

    if codes_ref.size(0) < sequence_length + codes_ref.size(0):
        codes_ref = F.pad(codes_ref, (0, sequence_length), value=config.datasets.audio_mask_token).unsqueeze(0)

    num_steps = nsf
    dt = 1.0 / num_steps
    x1_temp = 1.0
    gamma = 2.5
    mask_token_id = config.datasets.audio_mask_token
    S = vocab_size
    eps = 1e-9
    noise = 0.0

    mask_one_hot = torch.zeros((S), device=model.device)
    mask_one_hot[mask_token_id] = 1.0

    xt[..., : orig_ref_code_len] = codes_ref[..., : orig_ref_code_len]

    # Loop over the time grid
    for step in range(num_steps):
        t_val    = step * dt
        t_tensor = xt.new_full((1,), t_val)

        # unconditional pass
        logits_u = model(xt, text_ids, codes_ref, t_tensor, True,  True)
        probs_u  = torch.softmax(logits_u / x1_temp, -1)

        # conditional pass
        logits_c = model(xt, text_ids, codes_ref, t_tensor, False, False)
        probs_c  = torch.softmax(logits_c / x1_temp, -1)

        xt_mask  = (xt == mask_token_id).unsqueeze(-1).float()
        base_r   = (1 + noise * t_val) / (1 - t_val)

        R_u = xt_mask * probs_u * base_r
        R_c = xt_mask * probs_c * base_r

        remask = (1 - xt_mask) * mask_one_hot.view(1,1,S) * noise
        R_u += remask;  R_c += remask

        log_Ru = torch.log(R_u + eps)
        log_Rc = torch.log(R_c + eps)
        R_mix  = torch.exp(gamma * log_Rc + (1 - gamma) * log_Ru)

        # enforce row‑sum zero
        R_mix.scatter_(-1, xt[..., None], 0.)
        R_mix.scatter_(-1, xt[..., None], -R_mix.sum(-1, keepdim=True))

        # Euler step
        P = (R_mix * dt).clamp_min(0.)
        diag = (1. - P.sum(-1, keepdim=True)).clamp_min(0.)
        P.scatter_(-1, xt[..., None], diag)

        xt = torch.multinomial(P.view(-1, S), 1).view_as(xt)

        xt = apply_vlg_ops(
            x=xt,
            mask_token=config.datasets.audio_mask_token,
            expand_token=config.datasets.audio_expand_token,
            delete_token=config.datasets.audio_delete_token,
            max_len=config.test.max_audio_length,
            edit_start=orig_ref_code_len,
            edit_end=None
        )

        # codes_ref needs to have the same size as xt
        if codes_ref.size(1) < xt.size(1):
            codes_ref = F.pad(codes_ref, (0, xt.size(1) - codes_ref.size(1)), value=config.datasets.audio_mask_token)
        elif codes_ref.size(1) > xt.size(1):
            codes_ref = codes_ref[:, :xt.size(1)]

        xt[..., : orig_ref_code_len] = codes_ref[..., : orig_ref_code_len]

        # Optional: Corrector iterations at the current time step
        # for _ in range(num_corrector_steps):
        #     # Use a smaller corrector step (for example, 10% of h)
        #     h_corr = dt * 0.1
        #     logits_corr = model(xt, text_ids, codes_ref, t_tensor, False, False)
        #     p1_corr = torch.softmax(logits_corr, dim=-1)
        #     one_hot_xt_corr = torch.nn.functional.one_hot(xt, num_classes=vocab_size).float()

        #     # Compute the corrector velocity similarly
        #     u_corr = (p1_corr - one_hot_xt_corr) / (1.0 - t_val.item() + 1e-8)
        #     new_probs_corr = one_hot_xt_corr + h_corr * u_corr
        #     new_probs_corr = new_probs_corr / new_probs_corr.sum(dim=-1, keepdim=True)
        #     xt = torch.distributions.Categorical(probs=new_probs_corr).sample()

    return xt


@torch.no_grad()
def main() -> None:
    output_dir = "outputs_pfg_dynamic_dur_v2"
    gpu = 0
    config_path = "/raid/aluno_alef/DFM-TTS-2/config/default_offline_bpe_dynamic_dur_en.yaml"
    pretrained_checkpoint = "/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/qvrlk3in/checkpoints/epoch=24-step=422800-val/loss_epoch=3.265.ckpt"

    config = OmegaConf.load(config_path)

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    tokenizer = VoiceBpeTokenizer(vocab_file=config.datasets.vocab_file)
    model = DFMTTSWrapper.load_from_checkpoint(pretrained_checkpoint, config=config, map_location=device, strict=False)
    model.eval()

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

    # sentences = [
    #     "Embora estivesse chovendo, eles decidiram passear na floresta.",
    #     "Por causa do trânsito intenso, chegamos à reunião um pouco atrasados.",
    #     "Se você quer ter sucesso, deve estar preparado para trabalhar muito duro e manter o foco.",
    #     "Antes de sair de férias, lembre-se de regar as plantas e trancar todas as portas.",
    #     "Depois de terminar o trabalho, ele relaxou ouvindo música clássica e lendo um livro.",
    # ]

    nsf = [128, 256, 512, 1024, 2048]
    # nsf = [2048]

    os.makedirs(output_dir, exist_ok=True)

    for idx, sentence in enumerate(sentences):
        print(f"Processing sentence {idx + 1}/{len(sentences)}")
        for n in tqdm(nsf):
            text_ref = "He spoke with an extreme Oxford accent, and when he was talking well, his face sometimes wore the rapt expression of a very emotional man listening to music."
            x_t = inference(
                config=config,
                model=model,
                tokenizer=tokenizer,
                sentence=sentence,
                nsf=n,
                text_ref=text_ref,
                codes_ref=codes_ref,
                sequence_length=300,
                device=device
            )
            # remove making tokens from the generated sequence
            x_t = x_t.squeeze(0)
            x_t = x_t[codes_ref.size(0):]
            print("1", x_t.shape)
            x_t = x_t[x_t != config.datasets.audio_mask_token]
            print("2", x_t.shape)
            # remove padding tokens from the generated sequence
            x_t = x_t[x_t != config.datasets.audio_pad_token]
            print("3", x_t.shape)
            print(x_t)
            x_t = x_t[x_t != config.datasets.audio_expand_token]
            print("Shape after expand removal:", x_t.shape)
            x_t = x_t[x_t != config.datasets.audio_delete_token]
            x_t = x_t.unsqueeze(0).unsqueeze(0)
            print("4", x_t.shape)
            # Decode the final token sequence into an audio waveform
            generated_audio = audio_codec.decode_code(x_t)
            print("5", generated_audio.shape)
            torchaudio.save(f"{output_dir}/audio_{idx}-{n}.wav", generated_audio.squeeze(0).cpu(), 16000)


if __name__ == "__main__":
    main()
