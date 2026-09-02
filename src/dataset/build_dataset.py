import os
import operator
from typing import Tuple

from tqdm import tqdm
import pandas as pd
from datasets import Audio, load_dataset
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor

from utils.tokenizer import VoiceBpeTokenizer
from dataset.dataloader import (
    DynamicSingleSpeakerDataset,
    OfflineMultipleSpeakerDataset,
    HFTextTokenizerDataset,
    PhonemesDataset
)


def _has_non_empty_text(text) -> bool:
    """Row-level predicate used by IterableDataset.filter (runs lazily, in
    the DataLoader workers). Module-level so it pickles for spawn workers."""
    return text is not None and bool(str(text).strip())


def _apply_streaming_filters(dataset, config, split_name: str, row_filter=None):
    """Lazy, per-row filtering of a streamed IterableDataset.

    Nothing is materialized: the predicate is evaluated inside the DataLoader
    workers as rows are pulled from the Parquet shards.

    Config keys (all optional, under ``datasets``):

    filters:
        List of ``[column, op, value]`` triples in pyarrow DNF form, e.g.
        ``[[language, "==", pt], [stt_parakeet, "!=", ""]]``.
        These are pushed down into the Parquet reader (``load_dataset(...,
        filters=...)``) so rejected rows are dropped *before* the audio column
        is decoded. Comparisons against NULL evaluate to NULL and are dropped,
        so ``!= ""`` also removes missing values. Pushdown needs
        datasets>=2.19; older versions get the same predicates evaluated
        lazily in Python via ``row_filter`` (see ``_DNFRowFilter``).

    drop_empty_text (default: true):
        Additionally drop rows whose ``text_column`` is null, empty or
        whitespace-only, using ``IterableDataset.filter`` on that single
        column. Catches whitespace-only strings the Arrow filter lets through.
    """
    if row_filter is not None:
        dataset = dataset.filter(row_filter, input_columns=row_filter.columns)

    text_column = config.datasets.get("text_column", None)
    if config.datasets.get("drop_empty_text", True) and text_column:
        dataset = dataset.filter(
            _has_non_empty_text,
            input_columns=[text_column],
        )
        print(
            f"Streaming {split_name} dataset: dropping rows with empty "
            f"'{text_column}'"
        )
    return dataset


def _pushdown_filters(config):
    """Convert ``datasets.filters`` from the YAML into pyarrow DNF form:
    a list of AND-groups that are OR-ed together."""
    filters = config.datasets.get("filters", None)
    if not filters:
        return None
    filters = OmegaConf.to_container(filters, resolve=True)
    # Accept a flat list of triples (AND) or a list of lists (OR of ANDs).
    if filters and isinstance(filters[0][0], (list, tuple)):
        return [[tuple(f) for f in group] for group in filters]
    return [[tuple(f) for f in filters]]


def _parquet_builder_supports_filters() -> bool:
    """``ParquetConfig.filters`` (predicate pushdown) exists in datasets>=2.19."""
    try:
        from datasets.packaged_modules.parquet.parquet import ParquetConfig

        return "filters" in ParquetConfig.__dataclass_fields__
    except Exception:
        return False


class _DNFRowFilter:
    """Evaluate pyarrow-style DNF filters row by row in Python.

    Fallback for datasets versions without Parquet predicate pushdown. Same
    semantics as Arrow: any comparison involving a NULL is False, so
    ``!= ""`` drops missing values too. Picklable (module-level class) so it
    survives DataLoader worker spawn/fork.
    """

    _OPS = {
        "==": operator.eq,
        "=": operator.eq,
        "!=": operator.ne,
        "<": operator.lt,
        "<=": operator.le,
        ">": operator.gt,
        ">=": operator.ge,
        "in": lambda a, b: a in b,
        "not in": lambda a, b: a not in b,
    }

    def __init__(self, dnf):
        self.dnf = dnf
        self.columns = sorted({col for group in dnf for col, _, _ in group})
        for group in dnf:
            for _, op, _ in group:
                if op not in self._OPS:
                    raise ValueError(
                        f"Unsupported filter op '{op}'. "
                        f"Use one of: {sorted(self._OPS)}"
                    )

    def __call__(self, *values) -> bool:
        row = dict(zip(self.columns, values))
        for group in self.dnf:
            ok = True
            for col, op, value in group:
                actual = row[col]
                if actual is None or not self._OPS[op](actual, value):
                    ok = False
                    break
            if ok:
                return True
        return False


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

        load_kwargs = {}
        row_filter = None
        dnf = _pushdown_filters(config)
        if dnf is not None:
            if _parquet_builder_supports_filters():
                # Predicate pushdown into the Parquet reader: filtered-out
                # rows never reach Python nor the audio decoder.
                load_kwargs["filters"] = dnf
                print(f"Streaming Parquet row filters (pushdown): {dnf}")
            else:
                # Older `datasets`: evaluate the same predicates lazily with
                # IterableDataset.filter inside the DataLoader workers.
                row_filter = _DNFRowFilter(dnf)
                print(
                    "Streaming Parquet row filters (python fallback, "
                    f"datasets<2.19): {dnf}"
                )

        dataset = load_dataset(
            "parquet",
            data_files=data_files,
            streaming=True,
            **load_kwargs,
        )

        # Hand the raw encoded bytes to the collator instead of letting
        # `datasets` decode through torchcodec. The collator decodes with
        # soundfile and drops rows it cannot decode rather than crashing the
        # DataLoader worker (see StreamingHFTextTokenizerCollator).
        audio_column = config.datasets.get("audio_column", "audio")
        dataset = {
            split: ds.cast_column(audio_column, Audio(decode=False))
            for split, ds in dataset.items()
        }

        train_dataset = _apply_streaming_filters(
            dataset["train"], config, "training", row_filter
        )
        val_dataset = (
            _apply_streaming_filters(
                dataset["validation"], config, "validation", row_filter
            )
            if val_metadata
            else None
        )

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