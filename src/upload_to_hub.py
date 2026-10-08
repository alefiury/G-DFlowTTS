import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from omegaconf import OmegaConf
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import HfApi

from hub.configuration_gdflowtts import GDFlowTTSConfig
from hub.modeling_gdflowtts import GDFlowTTSModel


DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def build_hf_config(config) -> GDFlowTTSConfig:
    if config.get("scheduler_type", "polynomial") != "polynomial":
        raise ValueError(f"Unsupported scheduler_type: {config.scheduler_type}")
    if config.source_dist_type != "mask":
        raise ValueError(f"Unsupported source_dist_type: {config.source_dist_type}")
    if config.datasets.get("codec_name", "") != "neucodec":
        raise ValueError(f"Unsupported codec_name: {config.datasets.get('codec_name')}")

    m = config.model
    return GDFlowTTSConfig(
        audio_vocab_size=int(m.audio_vocab_size),
        text_vocab_size=int(m.text_vocab_size),
        hidden_size=int(m.hidden_size),
        cond_dim=int(m.cond_dim),
        n_heads=int(m.n_heads),
        n_blocks=int(m.n_blocks),
        dropout=float(m.dropout),
        audio_add_token=int(m.audio_add_token),
        text_add_token=int(m.text_add_token),
        text_filler_token=int(m.text_filler_token),
        audio_eos_token=int(config.datasets.audio_eos_token),
        audio_mask_token=int(config.datasets.audio_mask_token),
        scheduler_exponent=1.0,
        cond_drop_prob=float(config.datasets.get("cond_drop_prob", 0.0)),
        codec_name="neuphonic/neucodec",
    )


def load_model_state_dict(checkpoint_path: str) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    state_dict = ckpt.get("state_dict", ckpt)
    out = {}
    for key, value in state_dict.items():
        if not key.startswith("model."):
            continue  # e.g. audio_codec.* or other wrapper state
        key = key[len("model."):]
        if key.startswith("rotary_emb."):
            continue  # recomputed on the fly
        out[key] = value.clone()
    return out


@torch.no_grad()
def compare_with_original(hf_model: GDFlowTTSModel, config, state_dict: dict, device: torch.device) -> float:
    """Max abs logit difference between the original Transformer and the HF port."""
    from modules.gdflowtts.model import Transformer

    original = Transformer(**config.model)
    missing, unexpected = original.load_state_dict(state_dict, strict=False)
    assert not unexpected and all(k.startswith("rotary_emb.") for k in missing), (missing, unexpected)
    original.to(device).eval()

    g = torch.Generator().manual_seed(0)
    cfg = hf_model.config
    seq_len, text_len = 300, 60
    x_t = torch.randint(0, cfg.audio_vocab_size + cfg.audio_add_token, (2, seq_len), generator=g)
    x_t[:, 150:] = cfg.audio_mask_token
    text = torch.randint(0, cfg.text_vocab_size, (2, text_len), generator=g)
    time = torch.rand(2, generator=g)
    audio_mask = torch.ones(2, seq_len, dtype=torch.bool)
    audio_mask[1, 250:] = False
    x_t, text, time, audio_mask = (t.to(device) for t in (x_t, text, time, audio_mask))

    max_diff = 0.0
    for drop_text in (False, True):
        ref = original(x_t=x_t, text=text, time=time, drop_text=drop_text, audio_att_mask=audio_mask).float()
        out = hf_model(x_t=x_t, text_ids=text, time=time, drop_text=drop_text, audio_att_mask=audio_mask).float()
        max_diff = max(max_diff, (ref - out).abs().max().item())

    original.cpu()
    del original
    return max_diff


TRAINING_DATASET = {
    "name": "NeuCodec Emilia-YODAS (English)",
    "repo": "neuphonic/emilia-yodas-english-neucodec",
    "source": "amphion/Emilia-YODAS",
    "hours": "> 78k",
    "utterances": "> 30M",
    "language": "English",
    "license": "CC BY 4.0",
}

LOSS_NAMES = {"cross_entropy": "Cross-entropy", "generalized_kl": "Generalized KL"}


def write_model_card(output_dir: str, repo_id: str, config, args) -> None:
    ds = TRAINING_DATASET
    loss = config.get("loss", {}).get("function", "cross_entropy") if config.get("loss") else "cross_entropy"
    cond_drop_prob = float(config.datasets.get("cond_drop_prob", 0.0))

    badges = []
    if args.paper_url:
        arxiv_id = args.paper_url.rstrip("/").split("/")[-1]
        badges.append(f"[![arXiv](https://img.shields.io/badge/arXiv-{arxiv_id}-b31b1b.svg?logo=arxiv)]({args.paper_url})")
    if args.github_url:
        repo_name = args.github_url.rstrip("/").split("/")[-1].replace("-", "--")
        badges.append(f"[![GitHub](https://img.shields.io/badge/GitHub-{repo_name}-181717.svg?logo=github)]({args.github_url})")
    badges.append("[![SLT 2026](https://img.shields.io/badge/IEEE%20SLT-2026-00629B.svg)](https://attend.ieee.org/slt-2026/)")
    badges_line = "\n" + " ".join(badges) + "\n" if badges else ""
    license_line = f"license: {args.license}\n" if args.license else ""
    license_section = (
        f"## License\n\n"
        f"Released under the `{args.license}` license.\n"
        f"The training data ([`{ds['repo']}`](https://huggingface.co/datasets/{ds['repo']})) is {ds['license']}, "
        f"and [NeuCodec](https://huggingface.co/neuphonic/neucodec) remains subject to its own license.\n"
        if args.license
        else ""
    )
    pfg_note = (
        "The model was trained with text dropout, so prediction-free guidance (`use_pfg=True`) is supported."
        if cond_drop_prob > 0
        else "The model was trained without text dropout, so prediction-free guidance (`use_pfg`) is not available."
    )

    card = f"""---
{license_line}library_name: transformers
language:
- en
pipeline_tag: text-to-speech
datasets:
- {ds['repo']}
tags:
- audio
- speech
- text-to-speech
- tts
- zero-shot
- voice-cloning
- discrete-flow-matching
- neucodec
---

# Mask, Sample, Revise: A Revisable CTMC Inference Stack for Guided Discrete Flow Matching Text-to-Speech
{badges_line}
Accepted at the [IEEE Spoken Language Technology Workshop (SLT 2026)](https://attend.ieee.org/slt-2026/).

**G-DFlowTTS (Guided Discrete Flow Matching TTS)** is a zero-shot,
alignment-free text-to-speech model based on **Discrete Flow Matching (DFM)**.
A DiT with adaptive layer norm predicts
[NeuCodec](https://huggingface.co/neuphonic/neucodec) audio codes conditioned
on GPT-2 text tokens, and a Continuous-Time Markov Chain (CTMC) sampler infills
them in parallel after an acoustic prompt. This checkpoint was trained on
**{ds['name']}**.

In the paper we propose **Mask, Sample, Revise**, an inference-time CTMC stack
for DFM-TTS that requires no post-hoc fine-tuning:

- **SC-ReMask (Schedule-Constrained CTMC Remasking)**: a new remasking strategy
  for Discrete Flow Matching. It adapts the remasking formulation of
  [ReMDM](https://openreview.net/forum?id=IJryQAOy0p), originally designed for
  masked discrete diffusion, to the CTMC formulation of DFM. The two
  formulations are not directly compatible, so remasking is re-derived as an
  explicit token-to-mask CTMC transition whose rate is constrained by the
  probability-path schedule and added to the tau-leaping hazard. Generated
  tokens can return to the mask state and be revised, which yields good
  results with **fewer inference steps**.
- **Discrete guidance**: predictor-free guidance (PFG) mixes conditional and
  unconditional CTMC rates to strengthen text conditioning.
- **Prompt-matched conditional coupling**: training paths that match the
  prompted infilling task.

On LibriSpeech test-clean at 32 sampling steps, the full stack reduces WER
from 75.44% (unguided baseline) to 8.39%, outperforming unguided and
guidance-only samplers that use substantially more steps.

## Model

| | |
|---|---|
| **Architecture** | DiT (adaLN), {config.model.n_blocks} blocks, hidden size {config.model.hidden_size}, {config.model.n_heads} heads |
| **Parameters** | {args.n_params / 1e6:.0f}M |
| **Audio tokens** | [`neuphonic/neucodec`](https://huggingface.co/neuphonic/neucodec) (50 Hz, 65,536 codes) |
| **Text tokenizer** | [`{config.datasets.text_tokenizer_name}`](https://huggingface.co/{config.datasets.text_tokenizer_name}) |
| **Source distribution** | Masked |
| **Scheduler** | Polynomial (n = 1) |
| **Loss** | {LOSS_NAMES.get(loss, loss)} |
| **Text dropout** | {cond_drop_prob} |
| **Input sampling rate** | 16,000 Hz (reference audio) |
| **Output sampling rate** | 24,000 Hz |

## Training Dataset

| **Dataset** | **Hours** | **Utterances** | **Language** | **License** |
|---|---|---|---|---|
| [{ds['name']}](https://huggingface.co/datasets/{ds['repo']}) | {ds['hours']} | {ds['utterances']} | {ds['language']} | {ds['license']} |

The dataset contains the English subset of
[Emilia-YODAS](https://huggingface.co/datasets/{ds['source']}) pre-encoded with NeuCodec.

## Usage

Load the model directly from the Hugging Face Hub with Transformers. No clone
or manual snapshot download is required.

```python
import soundfile as sf
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "{repo_id}",
    trust_remote_code=True,
).to("cuda")

ref_audio, sr = sf.read("reference.wav")

wav = model.synthesize(
    text="I just received wonderful news about the promotion I have been waiting for!",
    ref_audio=ref_audio,
    ref_sampling_rate=sr,
    ref_text="Kids are talking by the door.",
    steps=32,
    # Predictor-free guidance
    use_pfg=True,
    gamma=1.5,
    # SC-ReMask: schedule-constrained CTMC remasking
    use_sc_remask=True,
    sc_remask_eta_rescale=0.5,
    sc_remask_eta_cap=0.5,
    sc_remask_tswitch=0.0,
    seed=0,
)

sf.write("output.wav", wav.numpy(), model.config.sampling_rate)
```

The settings above are the best configuration reported in the paper
(Mask, Sample, Revise): 32 sampling steps, PFG with γ = 1.5, and always-on
SC-ReMask (t_switch = 0, η_rescale = η_cap = 0.5).

`ref_text` must be the exact transcription of the reference audio. Reference
audio is converted to mono and resampled to **16,000 Hz** (resampling requires
`torchaudio`). The output length is estimated from the reference speaking rate;
use `speed` to adjust it or `duration` to set it explicitly in 50 Hz frames.

`trust_remote_code=True` is required because the G-DFlowTTS architecture and
sampler are shipped with this model repository. NeuCodec is loaded
automatically from its original Hugging Face repository.

### Sampling options

Extra keyword arguments of `synthesize` are forwarded to `model.sample_codes`:

| Argument | Default | Description |
|---|---|---|
| `steps` | 128 | Number of CTMC sampling steps |
| `speed` | 1.0 | Speaking rate used by the length heuristic |
| `duration` | `None` | Number of 50 Hz frames to generate (overrides `speed`) |
| `x1_temp`, `temp_schedule` | 1.0, `"dfm36"` | Sampling temperature and its schedule (`"dfm36"` or `"constant"`) |
| `use_pfg`, `gamma` | `False`, 1.5 | Prediction-free guidance |
| `use_tsr`, `tsr_k`, `tsr_sigma` | `False`, 1.0, 0.1 | Temporal score rescaling |
| `use_sc_remask` | `False` | SC-ReMask: schedule-constrained CTMC remasking (token-to-mask transitions that make generated tokens revisable) |
| `sc_remask_eta_rescale`, `sc_remask_eta_cap`, `sc_remask_tswitch` | 0.5, 0.5, 0.0 | SC-ReMask schedule: σ = η_rescale · min(η_cap, σ_max), disabled for t < t_switch |
| `sc_remask_use_conf`, `sc_remask_conf_threshold`, `sc_remask_beta`, `sc_remask_strength` | `False`, 0.35, 2.0, 1.0 | Optional confidence weighting: remask low-confidence tokens more |

{pfg_note}

The raw forward pass returns audio-code logits:
`model(x_t, text_ids, time, drop_text=False, audio_att_mask=None)`.

{license_section}"""
    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(card.rstrip() + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", required=True, help="Training YAML config used for the checkpoint.")
    parser.add_argument("-pc", "--checkpoint", required=True, help="Lightning .ckpt file.")
    parser.add_argument("-o", "--output-dir", required=True, help="Local directory for the exported repo.")
    parser.add_argument("--repo-id", default=None, help="Hub repo id, e.g. user/gdflowtts-en. Omit to only export.")
    parser.add_argument("--private", action="store_true", help="Create the Hub repo as private.")
    parser.add_argument("--dtype", default="float32", choices=list(DTYPES), help="dtype of the saved weights.")
    parser.add_argument("--skip-verify", action="store_true", help="Skip comparing logits with the original model.")
    parser.add_argument("--commit-message", default="Upload G-DFlowTTS model")
    parser.add_argument("--token", default=None, help="HF token (defaults to the cached login / HF_TOKEN).")
    parser.add_argument("--license", default="cc-by-nc-4.0", help="Model card license id; empty string to omit.")
    parser.add_argument("--paper-url", default="https://arxiv.org/abs/2606.13989", help="arXiv link for the model card badge; empty string to omit.")
    parser.add_argument("--github-url", default="https://github.com/alefiury/G-DFlowTTS", help="GitHub link for the model card badge; empty string to omit.")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Build the HF model and load the checkpoint weights
    hf_config = build_hf_config(config)
    hf_model = GDFlowTTSModel(hf_config)
    state_dict = load_model_state_dict(args.checkpoint)
    hf_model.load_state_dict(state_dict, strict=True)
    hf_model.eval()
    args.n_params = sum(p.numel() for p in hf_model.parameters())
    print(f"Loaded {len(state_dict)} tensors ({args.n_params / 1e6:.1f}M parameters)")

    if not args.skip_verify:
        hf_model.to(device)
        diff = compare_with_original(hf_model, config, state_dict, device)
        print(f"[verify] original vs HF port, max |logit diff| = {diff:.3e}")
        if diff > 1e-3:
            raise RuntimeError("HF port does not match the original model.")
        hf_model.cpu()

    # 2) Save safetensors + config + modeling code (auto_map for trust_remote_code)
    os.makedirs(args.output_dir, exist_ok=True)
    GDFlowTTSConfig.register_for_auto_class()
    GDFlowTTSModel.register_for_auto_class("AutoModel")
    hf_model.to(DTYPES[args.dtype]).save_pretrained(args.output_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(config.datasets.text_tokenizer_name)
    tokenizer.save_pretrained(args.output_dir)

    repo_id = args.repo_id or "<user>/<repo>"
    write_model_card(args.output_dir, repo_id, config, args)
    print(f"Exported to {args.output_dir}: {sorted(os.listdir(args.output_dir))}")

    # 3) Reload through AutoModel exactly as Hub users would
    if not args.skip_verify:
        reloaded = AutoModel.from_pretrained(args.output_dir, trust_remote_code=True, dtype=torch.float32)
        reloaded.to(device).eval()
        hf_model.to(device, torch.float32)
        g = torch.Generator().manual_seed(1)
        x_t = torch.randint(0, hf_config.audio_vocab_size, (1, 128), generator=g).to(device)
        text = torch.randint(0, hf_config.text_vocab_size, (1, 40), generator=g).to(device)
        time = torch.rand(1, generator=g).to(device)
        with torch.no_grad():
            diff = (reloaded(x_t, text, time) - hf_model(x_t, text, time)).abs().max().item()
        print(f"[verify] AutoModel reload ({args.dtype} weights) max |logit diff| = {diff:.3e}")
        if args.dtype == "float32" and diff > 1e-4:
            raise RuntimeError("Reloaded model does not match the exported model.")
        del reloaded

    # 4) Push to the Hub
    if args.repo_id:
        api = HfApi(token=args.token)
        api.create_repo(args.repo_id, private=args.private, exist_ok=True, repo_type="model")
        api.upload_folder(
            repo_id=args.repo_id,
            folder_path=args.output_dir,
            commit_message=args.commit_message,
            repo_type="model",
        )
        print(f"Uploaded to https://huggingface.co/{args.repo_id}")
    else:
        print("No --repo-id given; skipped upload.")


if __name__ == "__main__":
    main()
