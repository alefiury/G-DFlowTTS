import os
from glob import glob

import wandb
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm
import soundfile as sf

from xcodec2.modeling_xcodec2 import XCodec2Model


wandb.login()
def main():
    target_sr = 16000
    model_path = "HKUSTAudio/xcodec2"
    # output_dir = "/raid/aluno_alef/DATASETS/xcodec2/LibriTTS_R"
    # base_dir = "/raid/aluno_alef/DATASETS/LibriTTS_R"

    output_dir = "/raid/aluno_alef/DATASETS/xcodec2/CML"
    base_dir = "/raid/aluno_alef/DATASETS/CML"

    # output_dir = "/raid/aluno_alef/DATASETS/xcodec2/LJSpeech-1.1"
    # base_dir = "/raid/aluno_alef/DATASETS/LJSpeech-1.1"

    # output_dir = "/raid/aluno_alef/DATASETS/xcodec2/GigaSpeech"
    # base_dir = "/raid/aluno_alef/DATASETS/GigaSpeech"

    metadata_file = ""

    filepaths = glob(os.path.join(base_dir, "**", "*.wav"), recursive=True)
    filepaths += glob(os.path.join(base_dir, "**", "*.flac"), recursive=True)
    filepaths += glob(os.path.join(base_dir, "**", "*.mp3"), recursive=True)

    print(len(filepaths))

    # exit()

    wandb.init(project="Preprocess", entity="alefiury")
    wandb.run.name = "xcodec2_codes-libritts_r-b200"
    # wandb.run.save()

    model = XCodec2Model.from_pretrained(model_path)
    model.eval().cuda()

    for filepath in tqdm(filepaths, total=len(filepaths)):
        filename = os.path.basename(filepath)
        filedir = os.path.dirname(filepath)
        output_filepath = filedir.replace(base_dir, output_dir)
        output_filepath = os.path.join(output_filepath, filename)[:-4] + ".pt"

        # print(filepath)
        # print(output_filepath)

        try:
            if os.path.exists(output_filepath):
                continue

            wav, sr = torchaudio.load(filepath)
            if sr != target_sr:
                wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
                sr = target_sr

            with torch.no_grad():
                vq_code = model.encode_code(input_waveform=wav)
            #     recon_wav = model.decode_code(vq_code).cpu()       # Shape: (1, 1, T')

            #     sf.write("reconstructed.wav", recon_wav[0, 0, :].numpy(), sr)

            # break

            # print(filepath, output_filepath)
            # print(vq_code.shape)

            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
            torch.save(vq_code.cpu(), output_filepath)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue


if __name__ == "__main__":
    main()