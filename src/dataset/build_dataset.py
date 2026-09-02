import os
from typing import Tuple

from tqdm import tqdm
import pandas as pd
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor

from utils.tokenizer import VoiceBpeTokenizer
from dataset.dataloader import (
    DynamicSingleSpeakerDataset,
    OfflineMultipleSpeakerDataset,
    HFTextTokenizerDataset,
    PhonemesDataset
)


def build_dataset(config: dict) -> Tuple[DataLoader, DataLoader]:
    # Streaming Parquet path: keep the Hugging Face IterableDataset lazy and
    # avoid materializing metadata/audio in RAM.
    if config.datasets.type == "hf_streaming_text_tokenizer":
        val_metadata = config.datasets.get("val_metadata", "")

        data_files = {
            "train": config.datasets.train_metadata,
        }
        if val_metadata:
            data_files["validation"] = val_metadata

        dataset = load_dataset(
            "parquet",
            data_files=data_files,
            streaming=True,
        )

        train_dataset = dataset["train"]
        val_dataset = dataset["validation"] if val_metadata else None

        # torch DataLoader cannot randomly shuffle an IterableDataset.
        # Shuffle shards + a rolling example buffer here instead.
        if config.train.shuffle:
            train_dataset = train_dataset.shuffle(
                seed=config.datasets.get("shuffle_seed", 42),
                buffer_size=config.datasets.get("shuffle_buffer_size", 10_000),
            )

        train_shards = getattr(
            train_dataset, "num_shards", getattr(train_dataset, "n_shards", "?")
        )
        print(f"Streaming training dataset: {train_shards} shards")

        if val_dataset is not None:
            val_shards = getattr(
                val_dataset, "num_shards", getattr(val_dataset, "n_shards", "?")
            )
            print(f"Streaming validation dataset: {val_shards} shards")
        else:
            print(
                "No validation metadata configured for streaming training; "
                "the full training stream will be used for optimization."
            )

        return train_dataset, val_dataset

    train_df = pd.read_csv(config.datasets.train_metadata)

    if config.datasets.val_metadata == "":
        # Get 5000 random samples from train_df for validation
        val_df = train_df.sample(n=5000, random_state=42).reset_index(drop=True)
        # Remove validation samples from train_df
        train_df = train_df.drop(val_df.index).reset_index(drop=True)
    else:
        val_df = pd.read_csv(config.datasets.val_metadata)

    # if "language" column is not present, add it with a default value
    if "language" not in train_df.columns:
        train_df["language"] = "en"
    if "language" not in val_df.columns:
        val_df["language"] = "en"

    print(f"Number of training samples: {len(train_df)}")
    print(f"Number of validation samples: {len(val_df)}")


    if config.datasets.type == "dynamic":
        speech_processor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

        train_dataset = DynamicSingleSpeakerDataset(
            data=train_df,
            base_dir=config.datasets.base_dir,
            text_tokenizer=text_tokenizer,
            sampling_rate=config.datasets.sampling_rate,
            max_audio_duration=config.datasets.max_audio_duration,
            speech_processor=speech_processor,
        )

        val_dataset = DynamicSingleSpeakerDataset(
            data=val_df,
            base_dir=config.datasets.base_dir,
            text_tokenizer=text_tokenizer,
            sampling_rate=config.datasets.sampling_rate,
            max_audio_duration=config.datasets.max_audio_duration,
            speech_processor=speech_processor,
        )
    elif config.datasets.type == "offline" or \
        config.datasets.type == "offline_dynamic_dur" or \
        config.datasets.type == "offline_voice_cloning_simplified":
        text_tokenizer = VoiceBpeTokenizer(vocab_file=config.datasets.vocab_file)

        train_dataset = OfflineMultipleSpeakerDataset(
            data=train_df,
            base_dir=config.datasets.base_dir,
            filepath_column=config.datasets.filepath_column,
            text_tokenizer=text_tokenizer,
        )

        val_dataset = OfflineMultipleSpeakerDataset(
            data=val_df,
            base_dir=config.datasets.base_dir,
            filepath_column=config.datasets.filepath_column,
            text_tokenizer=text_tokenizer,
        )
    elif config.datasets.type == "hf_text_tokenizer":
        train_dataset = HFTextTokenizerDataset(
            data=train_df,
            base_dir=config.datasets.base_dir,
            filepath_column=config.datasets.filepath_column,
        )
        val_dataset = HFTextTokenizerDataset(
            data=val_df,
            base_dir=config.datasets.base_dir,
            filepath_column=config.datasets.filepath_column,
        )
    elif config.datasets.type == "phoneme_tokenizer":
        train_dataset = PhonemesDataset(
            data=train_df,
            base_dir=config.datasets.base_dir,
            filepath_column=config.datasets.filepath_column,
        )
        val_dataset = PhonemesDataset(
            data=val_df,
            base_dir=config.datasets.base_dir,
            filepath_column=config.datasets.filepath_column,
        )
    else:
        raise ValueError(f"Invalid dataset type: {config.datasets.type}")

    return train_dataset, val_dataset