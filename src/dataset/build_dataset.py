import os
from typing import Tuple

from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor

from utils.tokenizer import VoiceBpeTokenizer
from dataset.dataloader import DynamicSingleSpeakerDataset, OfflineMultipleSpeakerDataset


def build_dataset(config: dict) -> Tuple[DataLoader, DataLoader]:
    train_df = pd.read_csv(config.datasets.train_metadata)
    val_df = pd.read_csv(config.datasets.val_metadata)

    # if "language" column is not present, add it with a default value
    if "language" not in train_df.columns:
        train_df["language"] = "en"
    if "language" not in val_df.columns:
        val_df["language"] = "en"

    text_tokenizer = VoiceBpeTokenizer(vocab_file=config.datasets.vocab_file)

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
    elif config.datasets.type == "offline":
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
    else:
        raise ValueError(f"Invalid dataset type: {config.datasets.type}")

    return train_dataset, val_dataset