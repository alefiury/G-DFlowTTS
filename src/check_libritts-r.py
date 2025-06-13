import os
from glob import glob

from tqdm import tqdm
import pandas as pd


def libri_tts(root_path, meta_files=None, ignored_speakers=None, base_dir_to_remove: str = None):
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

                if base_dir_to_remove is not None:
                    filename = wav_file.replace(base_dir_to_remove, "")

                items.append({
                    "text": text.strip(),
                    "filepath": wav_file,
                    "filename": filename,
                    "speaker_name": f"LTTS_{speaker_name}",
                    "root_path": root_path,
                })

    print(f"Number of items (before file check): {len(items)}")
    new_items = []
    for item in items:
        if not os.path.exists(item["filepath"]):
            print(f"[!] wav file does not exist: {item['filepath']}")
        else:
            new_items.append(item)
    print(f"Number of items (after file check): {len(new_items)}")
    return new_items


def main():
    base_dir_to_remove = "/hadatasets/alef.ferreira/DATASETS/LibriTTS_R/"
    output_dir = "/hadatasets/alef.ferreira/DATASETS/LibriTTS_R_xcodec2"
    libritts_r_base_dir_100 = "/hadatasets/alef.ferreira/DATASETS/LibriTTS_R/train-clean-100/"
    libritts_r_base_dir_360 = "/hadatasets/alef.ferreira/DATASETS/LibriTTS_R/train-clean-360/"
    libritts_r_base_dir_500 = "/hadatasets/alef.ferreira/DATASETS/LibriTTS_R/train-other-500/"
    libritts_r_dev_dir = "/hadatasets/alef.ferreira/DATASETS/LibriTTS_R/dev-clean/"
    libritts_r_test_dir = "/hadatasets/alef.ferreira/DATASETS/LibriTTS_R/test-clean/"

    libri_tts_100 = libri_tts(libritts_r_base_dir_100, base_dir_to_remove=base_dir_to_remove)
    libri_tts_360 = libri_tts(libritts_r_base_dir_360, base_dir_to_remove=base_dir_to_remove)
    libri_tts_500 = libri_tts(libritts_r_base_dir_500, base_dir_to_remove=base_dir_to_remove)
    libri_tts_dev = libri_tts(libritts_r_dev_dir, base_dir_to_remove=base_dir_to_remove)
    libri_tts_test = libri_tts(libritts_r_test_dir, base_dir_to_remove=base_dir_to_remove)

    libri_tts_100_df = pd.DataFrame(libri_tts_100)
    libri_tts_360_df = pd.DataFrame(libri_tts_360)
    libri_tts_500_df = pd.DataFrame(libri_tts_500)
    libri_tts_dev_df = pd.DataFrame(libri_tts_dev)
    libri_tts_test_df = pd.DataFrame(libri_tts_test)

    print(libri_tts_100_df)
    print(libri_tts_360_df)
    print(libri_tts_500_df)
    print(libri_tts_dev_df)
    print(libri_tts_test_df)

    print(libri_tts_100_df["filename"])
    print(libri_tts_360_df["filename"])
    print(libri_tts_500_df["filename"])
    print(libri_tts_dev_df["filename"])
    print(libri_tts_test_df["filename"])

    print(f"Number of items in LibriTTS-100: {len(libri_tts_100_df)}")
    print(f"Number of items in LibriTTS-360: {len(libri_tts_360_df)}")
    print(f"Number of items in LibriTTS-500: {len(libri_tts_500_df)}")
    print(f"Number of items in LibriTTS-dev: {len(libri_tts_dev_df)}")
    print(f"Number of items in LibriTTS-test: {len(libri_tts_test_df)}")

    # concat both dataframes
    libri_tts_df = pd.concat([libri_tts_100_df, libri_tts_360_df], ignore_index=True)
    print(libri_tts_df)

    # concat both dataframes
    libri_tts_df = pd.concat([libri_tts_100_df, libri_tts_360_df, libri_tts_500_df], ignore_index=True)
    print(libri_tts_df)

    # save to csv
    libri_tts_df.to_csv(os.path.join(output_dir, "/hadatasets/alef.ferreira/DATASETS/LibriTTS_R/libri_tts-train-clean-960.csv"), index=False)
    libri_tts_dev_df.to_csv(os.path.join(output_dir, "/hadatasets/alef.ferreira/DATASETS/LibriTTS_R/libri_tts-dev-clean.csv"), index=False)
    libri_tts_test_df.to_csv(os.path.join(output_dir, "/hadatasets/alef.ferreira/DATASETS/LibriTTS_R/libri_tts-test-clean.csv"), index=False)


if __name__ == "__main__":
    main()
