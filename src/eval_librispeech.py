import os
import time
import argparse

import torch
import pandas as pd
import torchaudio
from tqdm import tqdm
from omegaconf import OmegaConf

from modules.wrappers.pl_wrapper import DFMTTSWrapper
from modules.wrappers.dp_wrapper import DurationPredictorWrapper
from utils.sampling import (
    seed_everything,
    parse_int_list,
    build_text_inputs,
    truncate_at_first_eos,
    load_codec,
    build_path_from_config,
    sample_mask_ctmc,
    extract_codes,
    load_codes,
)


def resolve_codes_path(filepath: str) -> str:
    # Metadata CSVs from the XCodec2 experiments point to .pt files; map them to the NeuCodec copies
    filepath = filepath.replace("/xcodec2/LibriSpeech-test-clean-filtered/", "/neucodec/LibriSpeech/")
    return filepath.replace(".pt", ".safetensors")


def load_row_codes(row, key_codes: str, key_audio: str, codec, audio_base_dir: str) -> torch.Tensor:
    if key_codes in row and not pd.isna(row[key_codes]):
        path = resolve_codes_path(str(row[key_codes]))
        assert os.path.exists(path), f"File not found: {path}"
        codes = load_codes(path)
    else:
        path = os.path.join(audio_base_dir, str(row[key_audio]))
        assert os.path.exists(path), f"File not found: {path}"
        codes = extract_codes(codec, path)
    return codes.reshape(-1).long()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Lightning .ckpt.")
    parser.add_argument("--metadata_csv", type=str, required=True, help="Metadata CSV (see module docstring).")
    parser.add_argument("--audio_base_dir", type=str, default="",
                        help="Base directory for target_filename/reference_filename (raw-audio CSVs).")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--nsf", type=str, default="32",
                        help="Comma-separated numbers of sampling steps, e.g. '4,8,16,32'")
    parser.add_argument("--use_oracle_length", action="store_true",
                        help="Use the ground-truth target length.")
    parser.add_argument("--oracle_add_eos", action="store_true",
                        help="Add +1 to the oracle length for the pinned EOS token.")

    # Duration model (optional)
    parser.add_argument("--duration_config", type=str, default=None)
    parser.add_argument("--duration_ckpt", type=str, default=None)

    # Sampling knobs
    parser.add_argument("--x1_temp", type=float, default=1.0)
    parser.add_argument("--temp_schedule", type=str, default="dfm36", choices=["dfm36", "constant"])
    parser.add_argument("--remask_noise", type=float, default=0.0,
                        help="Ad-hoc remasking noise (token->MASK). Keep 0 when using --use_sc_remask.")

    # TSR
    parser.add_argument("--use_tsr", action="store_true", help="Enable Temporal Score Rescaling (TSR).")
    parser.add_argument("--tsr_k", type=float, default=1.0,
                        help="Sharpening factor k (>1 sharper, <1 flatter).")
    parser.add_argument("--tsr_sigma", type=float, default=0.1,
                        help="TSR sigma parameter.")

    # PFG
    parser.add_argument("--use_pfg", action="store_true", help="Enable predictor-free guidance.")
    parser.add_argument("--gamma", type=float, default=1.5, help="PFG guidance strength.")

    # SC-ReMask
    parser.add_argument("--use_sc_remask", action="store_true",
                        help="Enable SC-ReMask (schedule-constrained CTMC remasking).")
    parser.add_argument("--sc_remask_eta_rescale", type=float, default=0.5,
                        help="SC-ReMask rescale factor (multiplies min(eta_cap, sigma_max)).")
    parser.add_argument("--sc_remask_eta_cap", type=float, default=0.5,
                        help="SC-ReMask cap before rescale: min(eta_cap, sigma_max).")
    parser.add_argument("--sc_remask_tswitch", type=float, default=0.0,
                        help="Switch time: sigma=0 for t < tswitch (t in [0,1]). 0 means always on.")
    parser.add_argument("--sc_remask_use_conf", action="store_true",
                        help="Enable confidence-based remasking (low-confidence tokens remask more).")
    parser.add_argument("--sc_remask_conf_threshold", type=float, default=0.35,
                        help="Confidence threshold: only tokens with p_cur < thr get remask weight.")
    parser.add_argument("--sc_remask_beta", type=float, default=2.0,
                        help="Sharpness of low-confidence weighting.")
    parser.add_argument("--sc_remask_strength", type=float, default=1.0,
                        help="Scale confidence weights (clamped to [0,1]).")

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
        if args.duration_config is not None and args.duration_ckpt is not None:
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

    has_codes = {"filepath_codec", "reference_codec"}.issubset(df.columns)
    has_audio = {"target_filename", "reference_filename"}.issubset(df.columns)
    for col in ["text", "ref_text"]:
        if col not in df.columns:
            raise ValueError(f"metadata_csv missing required column: {col}")
    if not (has_codes or has_audio):
        raise ValueError(
            "metadata_csv needs filepath_codec/reference_codec (codes) or "
            "target_filename/reference_filename (raw audio) columns"
        )

    eos_id = int(getattr(config.datasets, "audio_eos_token", -1))
    mask_id = int(getattr(config.datasets, "audio_mask_token", -1))

    rtf_tuples = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            text = str(row["text"])
            text_ref = str(row["ref_text"]) if not pd.isna(row["ref_text"]) else None

            codes_ref = load_row_codes(
                row, "reference_codec", "reference_filename", codec, args.audio_base_dir
            ).to(device)

            # text inputs
            text_ids, text_att_mask, _tok = build_text_inputs(config, text, text_ref, device)

            # Oracle length
            oracle_len = None
            if args.use_oracle_length:
                oracle_codes = load_row_codes(row, "filepath_codec", "target_filename", codec, args.audio_base_dir)
                oracle_len = int(oracle_codes.numel())
                if args.oracle_add_eos:
                    oracle_len += 1

            suffix_len = oracle_len if oracle_len is not None else 2048

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
                        use_tsr=bool(args.use_tsr),
                        tsr_k=float(args.tsr_k),
                        tsr_sigma=float(args.tsr_sigma),
                        use_sc_remask=bool(args.use_sc_remask),
                        sc_remask_eta_rescale=float(args.sc_remask_eta_rescale),
                        sc_remask_eta_cap=float(args.sc_remask_eta_cap),
                        sc_remask_tswitch=float(args.sc_remask_tswitch),
                        sc_remask_use_conf=bool(args.sc_remask_use_conf),
                        sc_remask_conf_threshold=float(args.sc_remask_conf_threshold),
                        sc_remask_beta=float(args.sc_remask_beta),
                        sc_remask_strength=float(args.sc_remask_strength),
                    )
                    end_time = time.time()
                    total_pred_time = end_time - start_time

                prefix_len = int(codes_ref.numel())
                gen = xt_full[0, prefix_len:].detach().cpu()

                # Truncate at first EOS, then drop any remaining MASK
                gen = truncate_at_first_eos(gen, eos_id)
                gen = gen[gen != mask_id]

                gen_for_codec = gen.to(device).unsqueeze(0).unsqueeze(0)  # [1,1,T]
                wav = codec.decode_code(gen_for_codec).detach()

                total_wav_length = wav.shape[-1] / saving_sr
                rtf = total_pred_time / total_wav_length if total_wav_length > 0 else float("inf")
                rtf_tuples.append((idx, steps, total_pred_time, total_wav_length, rtf))

                torchaudio.save(out_wav, wav.squeeze(0).cpu(), saving_sr)

        except Exception as e:
            print(f"[row {idx}] error: {e}")
            continue

    rtf_df = pd.DataFrame(rtf_tuples, columns=["idx", "steps", "total_pred_time", "total_wav_length", "rtf"])
    rtf_df.to_csv(os.path.join(args.output_dir, "rtf_results.csv"), index=False)


if __name__ == "__main__":
    main()
