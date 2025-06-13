import os
from glob import glob

import torch
from tqdm import tqdm
import torchaudio
import pandas as pd

from joblib import Parallel, delayed, parallel_backend
from tqdm_joblib import tqdm_joblib

from xcodec2.modeling_xcodec2 import XCodec2Model

import wandb


def libri_tts(root_path, meta_files=None, ignored_speakers=None):
    """https://ai.google/tools/datasets/libri-tts/"""
    items = []
    if not meta_files:
        meta_files = glob(f"{root_path}/**/*trans.tsv", recursive=True)
    else:
        if isinstance(meta_files, str):
            meta_files = [os.path.join(root_path, meta_files)]

    for meta_file in tqdm(meta_files, desc="Parsing meta files"):
        with open(meta_file, "r", encoding="utf-8") as ttf:
            for line in ttf:
                cols = line.split("\t")
                file_name = cols[0]
                speaker_name, chapter_id, *_ = cols[0].split("_")
                _root_path = os.path.join(root_path, f"{speaker_name}/{chapter_id}")
                wav_file = os.path.join(_root_path, file_name + ".wav")
                text = cols[2]
                # ignore speakers if needed
                if isinstance(ignored_speakers, list) and speaker_name in ignored_speakers:
                    continue
                items.append({
                    "text": text,
                    "audio_file": wav_file,
                    "speaker_name": f"LTTS_{speaker_name}",
                    "root_path": root_path,
                })

    print(f"Number of items (before file check): {len(items)}")
    new_items = []
    for item in items:
        if not os.path.exists(item["audio_file"]):
            print(f" [!] wav file does not exist: {item['audio_file']}")
        else:
            new_items.append(item)
    print(f"Number of items (after file check): {len(new_items)}")
    return new_items

wandb.login()
def main():
    n_jobs = -1
    target_sr = 16000
    model_path = "HKUSTAudio/xcodec2"
    output_dir = "/hadatasets/alef.ferreira/DATASETS/LibriTTS_R_xcodec2"
    libritts_r_base_dir = "/hadatasets/alef.ferreira/DATASETS/LibriTTS_R/train-other-500/"

    wandb.init(project="Preprocess", entity="alefiury")

    wandb.run.name = "xcodec2_codes"
    wandb.run.save()

    # Get metadata for all audio files
    meta_items = libri_tts(libritts_r_base_dir)

    rows = []
    for item in tqdm(meta_items, total=len(meta_items), desc="Building dataframe"):
        text = item["text"].strip()
        filepath = item["audio_file"]
        speaker_name = item["speaker_name"]
        filename = filepath.replace("/hadatasets/alef.ferreira/DATASETS/LibriTTS_R/", "")
        rows.append({
            "filename": filename,
            "filepath": filepath,
            "text": text,
            "speaker": speaker_name,
        })

    df = pd.DataFrame(rows)
    print(df)

    # Load the model once on GPU.
    model = XCodec2Model.from_pretrained(model_path)
    model.eval().cuda()

    # Define a function to process one audio file.
    def process_audio(row):
        output_filepath = os.path.join(output_dir, row["filename"])[:-4] + ".pt"
        if os.path.exists(output_filepath):
            return
        audio_path = row["filepath"]
        wav, sr = torchaudio.load(audio_path)
        if sr != target_sr:
            wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
        with torch.no_grad():
            vq_code = model.encode_code(input_waveform=wav)
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        torch.save(vq_code.cpu(), output_filepath)

    # Process the DataFrame rows in parallel with a progress bar.
    with parallel_backend("threading", n_jobs=n_jobs):
        with tqdm_joblib(tqdm(desc="Processing audio", total=len(df))):
            Parallel()(delayed(process_audio)(row) for _, row in df.iterrows())


if __name__ == "__main__":
    main()
