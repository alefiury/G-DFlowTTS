import os
import json
import logging
import argparse
import warnings
from pprint import pprint
from typing import Tuple
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
    sequence_length: int = 300,
    device: torch.device = torch.device("cuda")
) -> Tensor:
    if text_ref is not None:
        augmented_sentence = text_ref + ". " + sentence
    else:
        augmented_sentence = sentence
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
    num_corrector_steps = 1  # number of corrector iterations per predictor step

    # Create a time grid from 0 to 1 with (num_steps + 1) points
    t_init = 0.0
    t_final = 1.0
    time_grid = torch.linspace(t_init, t_final, num_steps + 1, device=device)

    sequence_length = get_remaining_duration(
        duration_model,
        text_ids=text_ids,
        codes_ref=codes_ref,
        device=device
    )

    # sequence_length += 50

    # sequence_length = 2048

    print("-"*100)
    print("Predicted sequence length:", sequence_length)
    print("codes_ref.size(0):", codes_ref.size(0))

    # Initialize x_t; for example, using the masked source
    xt = source_distribution.sample((1, sequence_length + codes_ref.size(0)), device=device)

    orig_ref_code_len = codes_ref.size(0)

    print("-"*100)
    print(xt.shape, codes_ref.shape)

    # x_t[:, : codes_ref.shape[-1]] = codes_ref

    # if codes_ref.size(0) < config.datasets.max_audio_length:
    #     codes_ref = F.pad(codes_ref, (0, config.datasets.max_audio_length - codes_ref.size(0)), value=config.datasets.audio_mask_token).unsqueeze(0)

    if codes_ref.size(0) < sequence_length + codes_ref.size(0):
        codes_ref = F.pad(codes_ref, (0, sequence_length), value=config.datasets.audio_mask_token).unsqueeze(0)

    # print(codes_ref)
    # print(codes_ref.shape)

    num_steps = nsf
    dt = 1.0 / num_steps
    x1_temp = 1.0
    guidance_scale = config.datasets.guidance_scale
    # gamma = config.datasets.guidance_scale
    gamma = 2.0
    mask_token_id = config.datasets.audio_mask_token
    S = vocab_size
    eps = 1e-9
    noise = 0.2

    mask_one_hot = torch.zeros((S), device=model.device)
    mask_one_hot[mask_token_id] = 1.0

    print("="*100)

    print(xt.shape, codes_ref.shape)
    xt[..., : orig_ref_code_len] = codes_ref[..., : orig_ref_code_len]
    print(xt.shape, codes_ref.shape)

    # print("="*100)
    # print(xt)
    # print(codes_ref)

    text_att_mask = text_ids.new_ones((1, text_ids.size(1)), dtype=torch.bool)

    # Loop over the time grid
    for step in range(num_steps):
        t_val    = step * dt
        t_tensor = xt.new_full((1,), t_val)

        # unconditional pass
        logits_u = model(
            x_t=xt,
            text_ids=text_ids,
            text_att_mask=text_att_mask,
            time=t_tensor,
            drop_text=True
        )
        probs_u  = torch.softmax(logits_u / x1_temp, -1)

        # conditional pass
        logits_c = model(
            x_t=xt,
            text_ids=text_ids,
            text_att_mask=text_att_mask,
            time=t_tensor,
            drop_text=False
        )
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

        xt[..., : orig_ref_code_len] = codes_ref[..., : orig_ref_code_len]

        # for _ in range(num_corrector_steps):
        #     # Use a smaller corrector step (for example, 10% of h)
        #     h_corr = dt * 0.1
        #     logits_corr = model(xt, text_ids, codes_ref, t_tensor, False, False)
        #     p1_corr = torch.softmax(logits_corr, dim=-1)
        #     one_hot_x_t_corr = torch.nn.functional.one_hot(xt, num_classes=vocab_size).float()

        #     # Compute the corrector velocity similarly
        #     u_corr = (p1_corr - one_hot_x_t_corr) / (1.0 - t_tensor.item() + 1e-8)
        #     new_probs_corr = one_hot_x_t_corr + h_corr * u_corr
        #     new_probs_corr = new_probs_corr / new_probs_corr.sum(dim=-1, keepdim=True)
        #     xt = torch.distributions.Categorical(probs=new_probs_corr).sample()

    return xt


@torch.no_grad()
def main() -> None:
    output_dir = "outputs_pfg_pred_dur_eos"
    gpu = 0
    config_path = "/raid/aluno_alef/DFM-TTS-2/config/offline-bpe-text_cfg-en.yaml"
    pretrained_checkpoint = "/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/m1ejk3am/checkpoints/epoch=29-step=500000-val/loss_epoch=3.366.ckpt"

    duration_pred_config_path = "/raid/aluno_alef/DFM-TTS-2/config/duration_predictor_bpe_en.yaml"
    duration_pred_pretrained_checkpoint = "/raid/aluno_alef/DFM-TTS-2/src/Duration-Predictor-DFM-TTS/61g87haf/checkpoints/epoch=10-step=138116-val/loss_epoch=4.717.ckpt"

    config = OmegaConf.load(config_path)
    duration_pred_config = OmegaConf.load(duration_pred_config_path)

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    tokenizer = VoiceBpeTokenizer(vocab_file=config.datasets.vocab_file)
    model = DFMTTSWrapper.load_from_checkpoint(pretrained_checkpoint, config=config, map_location=device, strict=False)
    model.eval()

    duration_model = DurationPredictorWrapper.load_from_checkpoint(
        duration_pred_pretrained_checkpoint,
        config=duration_pred_config,
        map_location=device,
        strict=False
    )
    duration_model.eval()

    audio_codec = XCodec2Model.from_pretrained(config.datasets.audio_codec).to(device)
    audio_codec.eval()

    ref_path = "/raid/time_voz/DATASETS_TTS/LibriTTS_R/dev-clean/1462/170138/1462_170138_000001_000004.wav"
    # ref_path = "/raid/aluno_alef/DATASETS/dataset_alc_48k_md5/bbd699/100/a600e123eb.wav"

    audio_ref, audio_ref_sr = torchaudio.load(ref_path)
    if audio_ref_sr != 16000:
        audio_ref = torchaudio.transforms.Resample(audio_ref_sr, 16000)(audio_ref)

    codes_ref = audio_codec.encode_code(input_waveform=audio_ref).squeeze()

    # sentences = [
    #     "Active artists always appreciate artistic achievements and applaud awesome artworks.",
    #     "Brave bakers boldly baked big batches of brownies in beautiful bakeries.",
    #     "Daring dancers dazzled during dynamic dance displays, drawing delighted crowds.",
    #     "Excited engineers eagerly enjoyed exploring enormous engineering exhibits.",
    #     "Friendly farmers faithfully fostered fields, favoring fruitful crops.",
    #     "Gallant gophers gracefully gambled golden gooseberries on grandiose glaciers.",
    #     "Happy hikers harmoniously hiked through hilly landscapes on heavenly holidays."
    # ]

    sentences = [
        "Now, as all books not primarily intended as picture-books consist principally of types composed to form letterpress",
        "There is no way, however, to eliminate the possibility of the fibers having come from another identical shirt, end quote",
        "Oswald's Marine training in marksmanship, his other rifle experience and his established familiarity with this particular weapon",
        "Nevertheless, she attempted to commit suicide by driving her nails, purposely left long, into her throat.",
        "His being was like an all sided lens concentrating all joys in the one heart of his consciousness.",
        "When he reached Mary's shop, he turned into the court to the kitchen door.",
    ]

    # sentences = [
    #     "Embora estivesse chovendo, eles decidiram passear na floresta.",
    #     "Por causa do trânsito intenso, chegamos à reunião um pouco atrasados.",
    #     "Se você quer ter sucesso, deve estar preparado para trabalhar muito duro e manter o foco.",
    #     "Antes de sair de férias, lembre-se de regar as plantas e trancar todas as portas.",
    #     "Depois de terminar o trabalho, ele relaxou ouvindo música clássica e lendo um livro.",
    # ]

    nsf = [256, 512, 1024]

    os.makedirs(output_dir, exist_ok=True)

    for idx, sentence in enumerate(sentences):
        print(f"Processing sentence {idx + 1}/{len(sentences)}")
        for n in tqdm(nsf):
            text_ref = "He spoke with an extreme Oxford accent, and when he was talking well, his face sometimes wore the rapt expression of a very emotional man listening to music."
            # text_ref = None
            x_t = inference(
                config=config,
                model=model,
                duration_model=duration_model,
                tokenizer=tokenizer,
                sentence=sentence,
                nsf=n,
                text_ref=text_ref,
                codes_ref=codes_ref,
                device=device
            )
            # remove making tokens from the generated sequence
            x_t = x_t.squeeze(0)
            x_t = x_t[codes_ref.size(0):]
            print("1", x_t.shape)
            print("/"*100)
            # count how many eos tokens there are in x_t
            # eos_count = (x_t == config.datasets.audio_eos_token).sum().item()
            # eos_index = (x_t == config.datasets.audio_eos_token).nonzero(as_tuple=True)[0]
            # print(x_t)
            # if eos_index.numel() > 0:
            #     x_t = x_t[..., :eos_index]
            # print(f"Number of EOS tokens: {eos_count}, first EOS token at index: {eos_index}")
            x_t = x_t[x_t != config.datasets.audio_eos_token]
            x_t = x_t[x_t != config.datasets.audio_mask_token]
            print("2", x_t.shape)
            # remove padding tokens from the generated sequence
            x_t = x_t[x_t != config.datasets.audio_pad_token]
            print("3", x_t.shape)
            print(x_t)
            x_t = x_t.unsqueeze(0).unsqueeze(0)
            print("4", x_t.shape)
            # Decode the final token sequence into an audio waveform
            generated_audio = audio_codec.decode_code(x_t)
            print("5", generated_audio.shape)
            torchaudio.save(f"{output_dir}/audio_{idx}-{n}.wav", generated_audio.squeeze(0).cpu(), 16000)


if __name__ == "__main__":
    main()
