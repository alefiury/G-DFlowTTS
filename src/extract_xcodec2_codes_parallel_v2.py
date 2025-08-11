import os
from glob import glob
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from xcodec2.modeling_xcodec2 import XCodec2Model
import wandb


class AudioDataset(Dataset):
    def __init__(self, filepaths, target_sr: int = 16_000):
        self.filepaths = filepaths
        self.target_sr = target_sr
        self._resamplers = {}

    def __len__(self):
        return len(self.filepaths)

    def _resample(self, wav, in_sr):
        key = (in_sr, self.target_sr)
        if key not in self._resamplers:
            self._resamplers[key] = torchaudio.transforms.Resample(
                in_sr, self.target_sr, dtype=wav.dtype
            )
        return self._resamplers[key](wav)

    def __getitem__(self, idx):
        try:
            fp = self.filepaths[idx]
            wav, sr = torchaudio.load(fp)
            if sr != self.target_sr:
                wav = self._resample(wav, sr)
            return fp, wav
        except Exception as e:
            print(f"Error loading file {self.filepaths[idx]}: {e}")
            new_index = (idx + 1) % len(self.filepaths)
            return self[new_index]


def collate_fn(batch):
    return batch[0]


# -------------------- 2.  Main script -------------------- #
def main():
    target_sr = 16_000
    model_path = "HKUSTAudio/xcodec2"

    base_dir = "/raid/aluno_alef/DATASETS/Common_Voice_17_0"
    output_dir = "/raid/aluno_alef/DATASETS/xcodec2/Common_Voice_17_0"

    exts = ("wav", "flac", "mp3")
    filepaths = [
        p for ext in exts
        for p in glob(os.path.join(base_dir, "**", f"*.{ext}"), recursive=True)
    ]
    print(f"Found {len(filepaths):,} audio files")

    # --- Weights & Biases (optional) --- #
    wandb.login()
    wandb.init(project="Preprocess", entity="alefiury", name="xcodec2_codes-Common_Voice_17_0")

    # --- Model --- #
    device = torch.device("cuda:0")
    model = XCodec2Model.from_pretrained(model_path).eval().to(device)

    # --- DataLoader --- #
    ds = AudioDataset(filepaths, target_sr)
    dl = DataLoader(
        ds,
        batch_size=1,           # <- one element per batch as requested
        num_workers=8,          # tune: usually #CPU‑cores or slightly less
        pin_memory=True,        # faster CPU→GPU copy
        prefetch_factor=4,
        collate_fn=collate_fn,
    )

    # --- Processing loop --- #
    for filepath, wav in tqdm(dl, total=len(ds)):
        # filepath is a Python str from worker; wav is a (1, T) tensor on CPU
        filepath = filepath if isinstance(filepath, str) else filepath[0]
        out_path = (
            Path(filepath).with_suffix(".pt")
            .as_posix()
            .replace(base_dir, output_dir, 1)
        )
        if os.path.exists(out_path):
            continue

        wav = wav.to(device, non_blocking=True)

        try:
            with torch.inference_mode():
                vq_code = model.encode_code(input_waveform=wav)

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save(vq_code.cpu(), out_path)
        except Exception as e:
            print(f"Error processing file {filepath}: {e}")
            continue

    print("✅ Done!")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")  # optional speed tweak
    main()
