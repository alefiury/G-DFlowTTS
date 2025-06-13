import os
from glob import glob

import torch
from tqdm import tqdm
import torchaudio
import soundfile as sf
import pandas as pd

import wandb

from xcodec2.modeling_xcodec2 import XCodec2Model


def libri_tts(root_path, meta_files=None, ignored_speakers=None):
    """https://ai.google/tools/datasets/libri-tts/"""
    items = []
    if not meta_files:
        meta_files = glob(f"{root_path}/**/*trans.tsv", recursive=True)
    else:
        if isinstance(meta_files, str):
            meta_files = [os.path.join(root_path, meta_files)]

    for meta_file in tqdm(meta_files):
        _meta_file = os.path.basename(meta_file).split(".")[0]
        with open(meta_file, "r", encoding="utf-8") as ttf:
            for line in ttf:
                cols = line.split("\t")
                file_name = cols[0]
                speaker_name, chapter_id, *_ = cols[0].split("_")
                _root_path = os.path.join(root_path, f"{speaker_name}/{chapter_id}")
                wav_file = os.path.join(_root_path, file_name + ".wav")
                text = cols[2]
                # ignore speakers
                if isinstance(ignored_speakers, list):
                    if speaker_name in ignored_speakers:
                        continue
                items.append(
                    {
                        "text": text,
                        "audio_file": wav_file,
                        "speaker_name": f"LTTS_{speaker_name}",
                        "root_path": root_path,
                    }
                )
    print(f"Number of items: {len(items)}")
    new_items = []
    for item in items:
        if not os.path.exists(item["audio_file"]):
            print(f" [!] wav files don't exist - {item['audio_file']}")
        else:
            new_items.append(item)
    items = new_items
    print(f"Number of items: {len(items)}")
    return items

wandb.login()
def main():
    target_sr = 16000
    model_path = "HKUSTAudio/xcodec2"
    output_dir = "/hadatasets/alef.ferreira/DATASETS/LibriTTS_R_xcodec2"
    # libritts_r_base_dir = "/hadatasets/alef.ferreira/DATASETS/LibriTTS_R/train-clean-100/"
    # libritts_r_base_dir = "/hadatasets/alef.ferreira/DATASETS/LibriTTS_R/train-clean-360"
    libritts_r_base_dir = "/hadatasets/alef.ferreira/DATASETS/LibriTTS_R/train-other-500/"

    # libri_tts
    meta_files = libri_tts(libritts_r_base_dir)

    wandb.init(project="Preprocess", entity="alefiury")

    wandb.run.name = "xcodec2_codes"
    wandb.run.save()

    rows = []

    for item in tqdm(meta_files, total=len(meta_files)):
        text = item["text"].strip()
        filepath = item["audio_file"]
        speaker_name = item["speaker_name"]

        filename = filepath.replace("/hadatasets/alef.ferreira/DATASETS/LibriTTS_R/", "")

        rows.append(
            {
                "filename": filename,
                "filepath": filepath,
                "text": text,
                "speaker": speaker_name,
            }
        )

    df = pd.DataFrame(rows)

    audio_path = df.iloc[0]["filepath"]

    print(df)

    model = XCodec2Model.from_pretrained(model_path)
    model.eval().cuda()

    for index, row in tqdm(df.iterrows(), total=len(df)):
        output_filepath = os.path.join(output_dir, row["filename"])[:-4] + ".pt"

        if os.path.exists(output_filepath):
            continue

        audio_path = row["filepath"]
        wav, sr = torchaudio.load(audio_path)
        if sr != target_sr:
            wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
            sr = target_sr

        with torch.no_grad():
            vq_code = model.encode_code(input_waveform=wav)

        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        torch.save(vq_code.cpu(), output_filepath)


if __name__ == "__main__":
    main()