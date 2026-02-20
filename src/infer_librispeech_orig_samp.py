import os
import time
import math
import argparse
import warnings
from dataclasses import dataclass
from typing import Optional, List, Tuple, Callable, Union

warnings.filterwarnings("ignore")

import torch
from torch import Tensor
import torch.nn.functional as F
import pandas as pd
import torchaudio
from safetensors.torch import load_file
from torchaudio import transforms as T
from omegaconf import OmegaConf

from transformers import AutoTokenizer

from tqdm import tqdm

from neucodec import NeuCodec
from xcodec2.modeling_xcodec2 import XCodec2Model

from flow_matching.utils import ModelWrapper
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.path.scheduler import PolynomialConvexScheduler
# from flow_matching.examples.text.logic.flow import get_source_distribution
from modules.flow import MaskedSourceDistribution, UniformSourceDistribution, get_source_distribution

from modules.pl_wrapper import DFMTTSWrapper
from modules.dp_wrapper import DurationPredictorWrapper
from utils.tokenizer import VoiceBpeTokenizer


class DFMTTSPosteriorNoCFG(ModelWrapper):
    """
    Vanilla adapter:
    Official MixtureDiscreteEulerSolver expects model(x=..., t=...) -> p_{1|t} in [0,1].
    Your DFMTTSWrapper returns logits, so we just softmax them.

    To keep prefix pinned + mask-source absorbing *without touching the solver*,
    we overwrite p_{1|t} with one-hot(x_t) on "frozen" positions.
    """

    def __init__(
        self,
        base_model,
        vocab_size: int,
        *,
        prefix_len: int,
        mask_id: int,
        freeze_prefix: bool = True,
        freeze_non_mask: bool = True,   # mask-source absorbing: only MASK can change
        freeze_eos: bool = True,
        eos_id: int = -1,
        ref_codes: Optional[Tensor] = None
    ):
        super().__init__(base_model)
        self.base_model = base_model
        self.S = int(vocab_size)
        self.prefix_len = int(prefix_len)
        self.mask_id = int(mask_id)
        self.freeze_prefix = bool(freeze_prefix)
        self.freeze_non_mask = bool(freeze_non_mask)
        self.freeze_eos = bool(freeze_eos)
        self.eos_id = int(eos_id)
        self.ref_codes = ref_codes

    @torch.no_grad()
    def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
        """
        Solver calls: self.model(x=x_t, t=t.repeat(B), **extras)
        extras should include: text_ids, text_att_mask, audio_att_mask
        """
        text_ids = extras["text_ids"]
        text_att_mask = extras["text_att_mask"]
        audio_att_mask = extras["audio_att_mask"]

        if self.ref_codes is not None:
            x[:, :self.prefix_len] = self.ref_codes

        logits = self.base_model(
            x_t=x,
            text_ids=text_ids,
            time=t,
            drop_text=False,              # <- always conditional, NO CFG branch
            text_att_mask=text_att_mask,
            audio_att_mask=audio_att_mask,
        )

        probs = torch.softmax(logits, dim=-1)

        return probs


@torch.inference_mode()
def sample_with_official_solver(
    *,
    config,
    model,
    text_ids: Tensor,
    text_att_mask: Tensor,
    codes_ref_1d: Tensor,
    suffix_len: int,
    steps: int,
    device: torch.device,
) -> Tensor:
    S = int(config.datasets.audio_vocab_size) + int(config.model.audio_add_token)

    mask_id = int(config.datasets.audio_mask_token)
    eos_id = int(getattr(config.datasets, "audio_eos_token", -1))

    prefix_len = int(codes_ref_1d.numel())
    T = prefix_len + int(suffix_len)

    audio_att_mask = torch.ones((1, T), dtype=torch.bool, device=device)

    source_distribution = get_source_distribution(
        source_distribution=config.source_dist_type,
        mask_token=mask_id,
        vocab_size=S,
    )

    path = MixtureDiscreteProbPath(
        scheduler=PolynomialConvexScheduler(n=1.0)
    )

    wrapped = DFMTTSPosteriorNoCFG(
        base_model=model,
        vocab_size=S,
        prefix_len=prefix_len,
        mask_id=mask_id,
        freeze_prefix=True,
        freeze_non_mask=True,
        freeze_eos=True,
        eos_id=eos_id,
        ref_codes=codes_ref_1d.unsqueeze(0),
    )

    solver = MixtureDiscreteEulerSolver(
        model=wrapped,
        path=path,
        vocabulary_size=S,
        source_distribution_p=source_distribution,
    )

    # Vanilla usage: step_size + time_grid=[0,1]
    step_size = 1.0 / int(steps)
    time_grid = torch.tensor([0.0, 1.0])

    # x_init: prefix + masked suffix (+ eos at end if you use it)
    # x_init = torch.full((1, T), mask_id, device=device, dtype=torch.long)
    x_init = source_distribution.sample(
        tensor_size=(1, T), device=device
    )
    x_init[:, :prefix_len] = codes_ref_1d.unsqueeze(0)
    # if eos_id >= 0:
    #     x_init[:, -1] = eos_id

    x_out = solver.sample(
        x_init=x_init,
        step_size=step_size,
        time_grid=time_grid,
        verbose=False,
        text_ids=text_ids,
        text_att_mask=text_att_mask,
        audio_att_mask=audio_att_mask,
    )
    return x_out



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
    augmented = (text_ref + " . " + sentence) if (text_ref is not None and len(text_ref) > 0) else sentence

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



def parse_int_list(s: str) -> List[int]:
    # e.g. "4,8,16,32" or "256"
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def truncate_at_first_eos(tokens_1d: torch.Tensor, eos_id: Optional[int]) -> torch.Tensor:
    if eos_id is None:
        return tokens_1d
    pos = (tokens_1d == eos_id).nonzero(as_tuple=False)
    if pos.numel() == 0:
        return tokens_1d
    first = int(pos[0].item())
    return tokens_1d[:first]


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

    parser.add_argument("--max_rows", type=int, default=-1)
    # parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    # seed_everything(args.seed)

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

    assert eos_id >= 0, "You must have an EOS token for suffix generation."
    assert mask_id >= 0, "You must have a MASK token for suffix generation."

    print("EOS ID:", eos_id)
    print("MASK ID:", mask_id)

    rtf_tuples = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # try:
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

        # Load reference codes (prefix)
        # codes_ref = torch.load(ref_filepath_codec).squeeze()
        # codes_ref = extract_codes(codec, reference_filepath).squeeze()
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
                start_time = time.time()
                xt_full = sample_with_official_solver(
                    config=config,
                    model=model,
                    text_ids=text_ids,
                    text_att_mask=text_att_mask,
                    codes_ref_1d=codes_ref,
                    suffix_len=suffix_len,
                    steps=steps,
                    device=device,
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

            # wav_ref = codec.decode_code(ref_gen.unsqueeze(0).unsqueeze(0)).detach()
            # out_gen_ref = os.path.join(args.output_dir, f"gen_ref_{idx}-nsf{steps}.wav")
            # torchaudio.save(out_gen_ref, wav_ref.squeeze(0).cpu(), saving_sr)

            # Save
            torchaudio.save(out_wav, wav.squeeze(0).cpu(), saving_sr)

        # except Exception as e:
        #     print(f"[row {idx}] error: {e}")
        #     continue

    # Save RTFs
    rtf_df = pd.DataFrame(rtf_tuples, columns=["idx", "steps", "total_pred_time", "total_wav_length", "rtf"])
    rtf_df.to_csv(os.path.join(args.output_dir, "rtf_results.csv"), index=False)


if __name__ == "__main__":
    main()
