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
    device: torch.device = torch.device("cuda"),
    x1_temp: float = 0.8,
    noise: float = 0.3,
    guidance_scale: float = 2.0,
    alpha_strength: float = 20.0,
    ref_language: str = "pt",
    text_language: str = "pt"
) -> Tensor:
    # misc numerics
    eps = 1e-12
    # --- text ids ---
    # augmented_sentence = (text_ref + ". " + sentence) if text_ref is not None else sentence
    # text_ids = torch.tensor(tokenizer.encode(augmented_sentence, lang=language)).unsqueeze(0).to(device)

    text_ids_ref = tokenizer.encode(text_ref, lang=ref_language)
    text_ids_text = tokenizer.encode(sentence, lang=text_language)

    print(text_ids_ref[:-1], text_ids_text)

    text_ids = torch.tensor(text_ids_ref[:-1] + text_ids_text).unsqueeze(0).to(device)

    max_length = config.datasets.max_audio_length
    vocab_size = config.datasets.audio_vocab_size + config.model.add_token

    # --- source dist (all MASK) ---
    source_distribution = MaskedSourceDistribution(mask_token=config.datasets.audio_mask_token)

    # --- scheduler hyperparams (with safe defaults) ---
    # Path (κ) scheduler
    kappa_a = float(getattr(getattr(config, "sampler", {}), "kappa_a", 0.0))
    kappa_b = float(getattr(getattr(config, "sampler", {}), "kappa_b", 2.0))  # paper's good text default
    # Corrector α schedule (optional)
    alpha_a = float(getattr(getattr(config, "sampler", {}), "corrector_a", 0.25))
    alpha_b = float(getattr(getattr(config, "sampler", {}), "corrector_b", 0.5))

    # duration for target length
    # sequence_length = get_remaining_duration(duration_model, text_ids=text_ids, codes_ref=codes_ref, device=device)
    # sequence_length = 2048

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

        # unconditional pass
        logits_u = model(
            x_t=xt,
            text_ids=text_ids,
            text_att_mask=text_att_mask,
            time=t_tensor,
            drop_text=True
        )
        probs_u = torch.softmax(logits_u / x1_temp, dim=-1)

        # conditional pass
        logits_c = model(
            x_t=xt,
            text_ids=text_ids,
            text_att_mask=text_att_mask,
            time=t_tensor,
            drop_text=False
        )
        probs_c = torch.softmax(logits_c / x1_temp, dim=-1)

        xt_mask = (xt == mask_token_id).unsqueeze(-1).float()

        # rate magnitude scaled by κ̇/(1-κ) (plus your small "noise" bias)
        denom = (1.0 - tau).clamp_min(1e-6)
        base_r = (1.0 + noise * float(tau)) * (float(kdot) / float(denom))

        # raw rates for PFG
        R_u = xt_mask * probs_u * base_r
        R_c = xt_mask * probs_c * base_r

        # tiny re-mask trick to keep some mass on MASK when token is known
        remask = (1 - xt_mask) * mask_one_hot.view(1, 1, S) * noise
        R_u = R_u + remask
        R_c = R_c + remask

        # predictor-free blend: R^{(γ)} = R_c^γ * R_u^{1-γ}
        log_Ru = torch.log(R_u + eps)
        log_Rc = torch.log(R_c + eps)
        R_mix = torch.exp(guidance_scale * log_Rc + (1.0 - guidance_scale) * log_Ru)  # PFG (exact)

        # Preserve support (no off-diagonal when not masked)
        R_mix = R_mix * xt_mask
        # set diagonal negative so rows sum to zero (rate matrix property)
        R_mix.scatter_(-1, xt[..., None], 0.0)
        R_mix.scatter_(-1, xt[..., None], -R_mix.sum(-1, keepdim=True))

        # Euler CTMC step: convert rates to transition probs for this step
        P = (R_mix * dt_lin).clamp_min(0.0)
        row_off = P.sum(-1, keepdim=True)
        diag = (1.0 - row_off).clamp_min(0.0)
        P.scatter_(-1, xt[..., None], diag)
        P = torch.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
        P = P / P.sum(-1, keepdim=True).clamp_min(1e-12)

        # sample next tokens
        xt = torch.multinomial(P.view(-1, S), 1).view_as(xt)
        xt[..., :orig_ref_code_len] = codes_ref[..., :orig_ref_code_len]

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
            u_corr = (p1_corr - one_hot_x_t_corr) / denom  # forward-time velocity ~ (p1 - δx)/(1-κ)
            u_corr = u_corr * float(kdot)                  # scale by κ̇

            new_probs_corr = one_hot_x_t_corr + h_corr * u_corr
            new_probs_corr = new_probs_corr.clamp_min(0.0)
            new_probs_corr = new_probs_corr / new_probs_corr.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            xt = torch.distributions.Categorical(probs=new_probs_corr).sample()
            xt[..., :orig_ref_code_len] = codes_ref[..., :orig_ref_code_len]

    return xt


@torch.no_grad()
def main() -> None:
    output_dir = "w1kigq88-multilingual-bpe-pfg-pt-eos_as_pad-pad_as_loss-cubic-corrector"
    # libri_speech_test_clean_metadata = "/raid/aluno_alef/DATASETS/LibriTTS_R-test-clean.csv"

    # df = pd.read_csv(libri_speech_test_clean_metadata)

    # # leave only one sample per "speaker"
    # print(f"Original dataset size: {len(df)}")
    # df = df.drop_duplicates(subset=["speaker"], keep="first").reset_index(drop=True)
    # print(f"Reduced dataset size: {len(df)}")

    # print(df.columns)
    gpu = 0

    config_path = "/raid/aluno_alef/DFM-TTS-2/config/offline-bpe-text_cfg-eos_as_pad-multilingual.yaml"
    pretrained_checkpoint = "/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/w1kigq88/checkpoints/epoch=08-step=300000-val/loss_epoch=1.615.ckpt"

    config = OmegaConf.load(config_path)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    tokenizer = VoiceBpeTokenizer(vocab_file=config.datasets.vocab_file)
    model = DFMTTSWrapper.load_from_checkpoint(pretrained_checkpoint, config=config, map_location=device, strict=False)
    model.eval()

    audio_codec = XCodec2Model.from_pretrained(config.datasets.audio_codec).to(device)
    audio_codec.eval()

    nsf = [512, 1024]

    os.makedirs(output_dir, exist_ok=True)

    sentences = [
        ("pt", "Embora estivesse chovendo, eles decidiram passear na floresta."),
        ("pt", "Por causa do trânsito intenso, chegamos à reunião um pouco atrasados."),
        ("pt", "Mesmo se sentindo cansada, ela continuou a trabalhar em seu projeto até tarde da noite."),
        ("pt", "Se você quer ter sucesso, deve estar preparado para trabalhar muito duro e manter o foco."),
        ("pt", "Antes de sair de férias, lembre-se de regar as plantas e trancar todas as portas."),

        ("en", "Although it was raining, they decided to go for a walk in the forest."),
        ("en", "Because of the heavy traffic, we arrived at the meeting slightly late."),
        ("en", "Even though she felt tired, she continued to work on her project until late night."),
        ("en", "If you want to succeed, you must be prepared to work very hard and stay focused."),
        ("en", "Before leaving for vacation, remember to water the plants and lock all the doors."),

        ("it", "Sebbene piovesse, hanno deciso di fare una passeggiata nella foresta."),
        ("it", "A causa del traffico intenso, siamo arrivati alla riunione leggermente in ritardo."),
        ("it", "Anche se si sentiva stanca, ha continuato a lavorare al suo progetto fino a tarda notte."),
        ("it", "Se vuoi avere successo, devi essere pronto a lavorare molto duramente e rimanere concentrato."),
        ("it", "Prima di partire per le vacanze, ricordati di innaffiare le piante e chiudere a chiave tutte le porte."),

        ("pl", "Chociaż padało, postanowili wybrać się na spacer po lesie."),
        ("pl", "Z powodu dużego natężenia ruchu drogowego dotarliśmy na spotkanie nieco spóźnieni."),
        ("pl", "Mimo że była zmęczona, pracowała nad swoim projektem do późna w nocy."),
        ("pl", "Jeśli chcesz odnieść sukces, musisz być gotowy ciężko pracować i pozostać skupionym."),
        ("pl", "Przed wyjazdem na wakacje pamiętaj, by podlać rośliny i zamknąć wszystkie drzwi na klucz."),

        ("es", "Aunque llovía, decidieron dar un paseo por el bosque."),
        ("es", "Debido al intenso tráfico, llegamos un poco tarde a la reunión."),
        ("es", "A pesar de que se sentía cansada, siguió trabajando en su proyecto hasta altas horas de la noche."),
        ("es", "Si quieres tener éxito, debes estar dispuesto a trabajar muy duro y mantener la concentración."),
        ("es", "Antes de salir de vacaciones, recuerda regar las plantas y cerrar con llave todas las puertas."),

        ("fr", "Malgré la pluie, ils ont décidé de faire une promenade en forêt."),
        ("fr", "À cause de la circulation dense, nous sommes arrivés légèrement en retard à la réunion."),
        ("fr", "Même si elle était fatiguée, elle a continué à travailler sur son projet jusqu’à tard dans la nuit."),
        ("fr", "Si tu veux réussir, tu dois être prêt à travailler très dur et à rester concentré."),
        ("fr", "Avant de partir en vacances, n’oublie pas d’arroser les plantes et de fermer toutes les portes à clé."),
    ]
    text_ref = "A união faz a força, já dizia minha avó. E quando a gente se junta para resolver os problemas do bairro, as coisas fluem melhor."
    ref_filepath = "ermis_11labs-0000-0003.wav"

    # print(f"Processing {text}")

    audio_ref, audio_ref_sr = torchaudio.load(ref_filepath)
    if audio_ref_sr != 16000:
        audio_ref = torchaudio.transforms.Resample(audio_ref_sr, 16000)(audio_ref)
    codes_ref = audio_codec.encode_code(input_waveform=audio_ref).squeeze()

    # save audio ref
    # torchaudio.save(f"{output_dir}/audio_ref_{idx}.mp3", audio_ref.cpu(), 16000)

    for idx, (language, text) in tqdm(enumerate(sentences), total=len(sentences)):
        for n in tqdm(nsf):
            x_t = inference(
                config=config,
                model=model,
                duration_model=None,
                tokenizer=tokenizer,
                sentence=text,
                nsf=n,
                text_ref=text_ref,
                codes_ref=codes_ref,
                sequence_length=512,
                device=device,
                ref_language="pt",
                text_language=language
            )
            # remove making tokens from the generated sequence
            x_t = x_t.squeeze(0)
            x_t = x_t[codes_ref.size(0):]

            x_t = x_t[x_t != config.datasets.audio_eos_token]
            x_t = x_t[x_t != config.datasets.audio_mask_token]
            # remove padding tokens from the generated sequence
            if hasattr(config.datasets, "audio_pad_token"):
                x_t = x_t[x_t != config.datasets.audio_pad_token]
            x_t = x_t.unsqueeze(0).unsqueeze(0)
            # Decode the final token sequence into an audio waveform
            generated_audio = audio_codec.decode_code(x_t)
            torchaudio.save(f"{output_dir}/{idx}_audio_{n}.wav", generated_audio.squeeze(0).cpu(), 16000)


if __name__ == "__main__":
    main()
