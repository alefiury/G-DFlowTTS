import os
import torch
import pandas as pd
from tqdm import tqdm
import concurrent.futures

def process_file(row, base_dir):
    # Extract filename and build the proper filepath.
    filename = row["filename"]
    filepath = os.path.join(base_dir, filename)
    if filepath.endswith(".wav"):
        filepath = filepath[:-4] + ".pt"

    # Check file existence.
    if not os.path.exists(filepath):
        return f"File not found: {filepath}"

    # Try loading the tensor.
    try:
        tensor = torch.load(filepath)
        # print(tensor.shape)
    except Exception as e:
        return f"Error loading file: {filepath}\n{e}"

    # Return None if everything went fine.
    return None

def process_metadata(metadata, base_dir):
    # Convert dataframe rows to a list of dictionaries.
    rows = metadata.to_dict('records')
    errors = []

    # Create a process pool to handle the tasks concurrently.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Pass the base_dir for every row.
        results = executor.map(process_file, rows, [base_dir] * len(rows))
        for error in tqdm(results, total=len(rows), desc="Processing files"):
            if error:
                print(error)
                errors.append(error)
    return errors

def main():
    base_dir = "/hadatasets/alef.ferreira/DATASETS/LibriTTS_R_xcodec2"
    train_metadata_path = "/hadatasets/alef.ferreira/DATASETS/LibriTTS_R/libri_tts-train-clean-460.csv"
    dev_metadata_path = "/hadatasets/alef.ferreira/DATASETS/LibriTTS_R/libri_tts-dev-clean.csv"
    test_metadata_path = "/hadatasets/alef.ferreira/DATASETS/LibriTTS_R/libri_tts-test-clean.csv"

    # Read metadata CSV files.
    train_metadata = pd.read_csv(train_metadata_path)
    dev_metadata = pd.read_csv(dev_metadata_path)
    test_metadata = pd.read_csv(test_metadata_path)

    print("Train metadata columns:", train_metadata.columns)
    print(train_metadata)
    print(dev_metadata)
    print(test_metadata)

    # Process each metadata DataFrame in parallel.
    for metadata in [train_metadata, dev_metadata, test_metadata]:
        print(f"Checking {len(metadata)} files")
        process_metadata(metadata, base_dir)

if __name__ == "__main__":
    main()
