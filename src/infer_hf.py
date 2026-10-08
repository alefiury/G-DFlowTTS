import os
import time
import argparse

import torch
import soundfile as sf
from transformers import AutoModel

DEFAULT_MODEL = "alefiury/G-DFlowTTS-NeuCodec-Emilia-YODAS"
DEFAULT_REF_AUDIO = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "refs", "1462_170138_000001_000004.wav"
)
DEFAULT_REF_TEXT = (
    "He spoke with an extreme Oxford accent, and when he was talking well, his face sometimes "
    "wore the rapt expression of a very emotional man listening to music."
)
CODEC_FPS = 50  # NeuCodec tokens per second


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Hub repo id or local exported folder.")
    parser.add_argument("--text", type=str,
                        default="When he reached Mary's shop, he turned into the court to the kitchen door.",
                        help="Target text to synthesize.")
    parser.add_argument("--ref_audio", type=str, default=DEFAULT_REF_AUDIO, help="Audio prompt (any sample rate).")
    parser.add_argument("--ref_text", type=str, default=DEFAULT_REF_TEXT, help="Exact transcript of the audio prompt.")
    parser.add_argument("--output", type=str, default="outputs/sample.wav", help="Output wav path.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)

    # Length of the generated speech
    parser.add_argument("--duration", type=float, default=None,
                        help="Target duration in seconds. Default: estimated from the prompt's speaking rate.")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Speaking-rate factor for the length estimate (>1 faster, <1 slower).")

    # Mask, Sample, Revise sampler (paper defaults)
    parser.add_argument("--nfe", type=int, default=32, help="Number of sampling steps.")
    parser.add_argument("--gamma", type=float, default=1.5, help="Predictor-free guidance strength.")
    parser.add_argument("--no_pfg", action="store_true", help="Disable predictor-free guidance.")
    parser.add_argument("--no_remask", action="store_true", help="Disable SC-ReMask.")
    parser.add_argument("--eta_rescale", type=float, default=0.5, help="SC-ReMask rescale factor.")
    parser.add_argument("--eta_cap", type=float, default=0.5, help="SC-ReMask cap.")
    parser.add_argument("--t_switch", type=float, default=0.0, help="SC-ReMask switch time.")
    args = parser.parse_args()

    model = AutoModel.from_pretrained(args.model, trust_remote_code=True).to(args.device).eval()

    ref_audio, ref_sr = sf.read(args.ref_audio, dtype="float32")
    if ref_audio.ndim > 1:
        ref_audio = ref_audio.mean(axis=1)

    duration = None if args.duration is None else max(1, round(args.duration * CODEC_FPS))

    start = time.time()
    wav = model.synthesize(
        text=args.text,
        ref_audio=ref_audio,
        ref_text=args.ref_text,
        ref_sampling_rate=ref_sr,
        steps=args.nfe,
        speed=args.speed,
        duration=duration,
        seed=args.seed,
        use_pfg=not args.no_pfg,
        gamma=args.gamma,
        use_sc_remask=not args.no_remask,
        sc_remask_eta_rescale=args.eta_rescale,
        sc_remask_eta_cap=args.eta_cap,
        sc_remask_tswitch=args.t_switch,
    )
    elapsed = time.time() - start

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    sr = model.config.sampling_rate
    sf.write(args.output, wav.numpy(), sr)
    seconds = wav.shape[-1] / sr
    print(f"Saved {args.output} ({seconds:.2f}s, RTF {elapsed / max(seconds, 1e-6):.3f})")


if __name__ == "__main__":
    main()
