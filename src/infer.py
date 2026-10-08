import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torchaudio
from omegaconf import OmegaConf

from modules.wrappers.pl_wrapper import DFMTTSWrapper
from utils.sampling import (
    seed_everything,
    build_text_inputs,
    truncate_at_first_eos,
    load_codec,
    build_path_from_config,
    sample_mask_ctmc,
)

DEFAULT_REF_AUDIO = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "refs", "1462_170138_000001_000004.wav"
)
DEFAULT_REF_TEXT = (
    "He spoke with an extreme Oxford accent, and when he was talking well, his face sometimes "
    "wore the rapt expression of a very emotional man listening to music."
)
CODEC_FPS = 50  # NeuCodec tokens per second


def load_prompt_audio(filepath: str, target_sr: int = 16_000) -> torch.Tensor:
    """Load a prompt as a mono 16 kHz waveform shaped [1, 1, T]."""
    wav, sr = torchaudio.load(filepath)
    wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.unsqueeze(0)


def estimate_num_codes(ref_text: str, text: str, num_ref_codes: int, speed: float = 1.0) -> int:
    """Target length from the prompt's speaking rate (codes per character)."""
    return max(1, int(num_ref_codes / max(1, len(ref_text)) * len(text) / speed))


def main():
    parser = argparse.ArgumentParser(description="Synthesize one utterance from an audio prompt.")
    parser.add_argument("--config", type=str, required=True, help="Path to the training YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the Lightning .ckpt.")
    parser.add_argument("--text", type=str,
                        default="When he reached Mary's shop, he turned into the court to the kitchen door.",
                        help="Target text to synthesize.")
    parser.add_argument("--ref_audio", type=str, default=DEFAULT_REF_AUDIO, help="Audio prompt (any sample rate).")
    parser.add_argument("--ref_text", type=str, default=DEFAULT_REF_TEXT, help="Transcript of the audio prompt.")
    parser.add_argument("--output", type=str, default="outputs/sample.wav", help="Output wav path.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    # Length of the generated speech
    parser.add_argument("--duration", type=float, default=None,
                        help="Target duration in seconds. Default: estimated from the prompt's speaking rate.")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Speaking-rate factor for the length estimate (>1 faster, <1 slower).")

    # Mask, Sample, Revise sampler
    parser.add_argument("--nfe", type=int, default=32, help="Number of sampling steps.")
    parser.add_argument("--gamma", type=float, default=1.5, help="Predictor-free guidance strength.")
    parser.add_argument("--no_pfg", action="store_true", help="Disable predictor-free guidance.")
    parser.add_argument("--no_remask", action="store_true", help="Disable SC-ReMask.")
    parser.add_argument("--eta_rescale", type=float, default=0.5, help="SC-ReMask rescale factor.")
    parser.add_argument("--eta_cap", type=float, default=0.5, help="SC-ReMask cap.")
    parser.add_argument("--t_switch", type=float, default=0.0, help="SC-ReMask switch time.")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    config = OmegaConf.load(args.config)

    model = DFMTTSWrapper.load_from_checkpoint(
        args.checkpoint,
        config=config,
        map_location=device,
        strict=False,
        weights_only=False,
    ).to(device)
    model.eval()
    codec, saving_sr = load_codec(config, device)
    path = build_path_from_config(config)

    # Prompt codes and text (prompt transcript + target text)
    ref_codes = codec.encode_code(load_prompt_audio(args.ref_audio)).reshape(-1).to(device)
    text_ids, text_att_mask, _ = build_text_inputs(config, args.text, args.ref_text, device)

    if args.duration is not None:
        num_codes = max(1, round(args.duration * CODEC_FPS))
    else:
        num_codes = estimate_num_codes(args.ref_text, args.text, ref_codes.numel(), args.speed)
    # +1 for the EOS token pinned at the last position
    suffix_len = num_codes + 1

    print(f"Prompt: {ref_codes.numel() / CODEC_FPS:.2f}s | target: {num_codes / CODEC_FPS:.2f}s | NFE: {args.nfe}")

    start = time.time()
    with torch.no_grad():
        xt = sample_mask_ctmc(
            config=config,
            model=model,
            path=path,
            text_ids=text_ids,
            text_att_mask=text_att_mask,
            codes_ref_1d=ref_codes,
            suffix_len=suffix_len,
            steps=args.nfe,
            device=device,
            x1_temp=1.0,
            temp_schedule="dfm36",
            remask_noise=0.0,
            use_pfg=not args.no_pfg,
            gamma=args.gamma,
            use_tsr=False,
            use_sc_remask=not args.no_remask,
            sc_remask_eta_rescale=args.eta_rescale,
            sc_remask_eta_cap=args.eta_cap,
            sc_remask_tswitch=args.t_switch,
            sc_remask_use_conf=False,
        )
    elapsed = time.time() - start

    # Keep the generated suffix, cut at the first EOS and drop leftover masks
    gen = xt[0, ref_codes.numel():].cpu()
    gen = truncate_at_first_eos(gen, int(config.datasets.audio_eos_token))
    gen = gen[gen != int(config.datasets.audio_mask_token)]
    wav = codec.decode_code(gen.to(device)[None, None, :]).squeeze(0).cpu()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torchaudio.save(args.output, wav, saving_sr)
    duration = wav.shape[-1] / saving_sr
    print(f"Saved {args.output} ({duration:.2f}s, RTF {elapsed / max(duration, 1e-6):.3f})")


if __name__ == "__main__":
    main()
