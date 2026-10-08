import os
import glob
import fnmatch
import operator
from typing import Tuple

from tqdm import tqdm
import pandas as pd
from datasets import Audio, load_dataset
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dataset.dataloader import (
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


def _streaming_load_kwargs(config):
    """Translate ``datasets.filters`` into ``load_dataset`` kwargs (Parquet
    predicate pushdown) or, for older ``datasets``, into a lazy row filter."""
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
    return load_kwargs, row_filter


def _finalize_streaming_splits(train_dataset, val_dataset, config):
    """Shuffle the training stream and log the shard layout."""
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
            "No validation data configured for streaming training; "
            "the full training stream will be used for optimization."
        )

    return train_dataset, val_dataset


def _list_codes_shards(config):
    """Sorted Parquet shards of a codes dataset, from the Hugging Face Hub
    (``hf_dataset_name``) or from a local glob (``train_metadata``)."""
    if config.datasets.get("hf_dataset_name", ""):
        from huggingface_hub import HfApi

        pattern = config.datasets.get("hf_data_files", "data/train-*.parquet")
        files = HfApi().list_repo_files(
            config.datasets.hf_dataset_name,
            repo_type="dataset",
            revision=config.datasets.get("hf_revision", None),
        )
        return sorted(f for f in files if fnmatch.fnmatch(f, pattern))
    return sorted(glob.glob(config.datasets.train_metadata))


def _build_streaming_codes_dataset(config):
    """Stream a dataset whose rows already hold codec tokens, e.g.
    ``neuphonic/emilia-yodas-english-neucodec`` (``text`` + ``codes``).

    Validation is drawn from held-out shards (the last ``val_num_shards``),
    so no training row is ever used for validation and the training stream
    keeps shard-level shuffling.
    """
    shards = _list_codes_shards(config)
    if not shards:
        raise ValueError("No Parquet shards found for the streaming codes dataset.")

    val_num_samples = int(config.datasets.get("val_num_samples", 0))
    val_num_shards = int(config.datasets.get("val_num_shards", 1)) if val_num_samples > 0 else 0
    if val_num_shards >= len(shards):
        raise ValueError(
            f"val_num_shards={val_num_shards} leaves no training shards "
            f"(found {len(shards)})."
        )

    data_files = {"train": shards[: len(shards) - val_num_shards]}
    if val_num_shards > 0:
        data_files["validation"] = shards[len(shards) - val_num_shards:]

    load_kwargs, row_filter = _streaming_load_kwargs(config)
    dataset = load_dataset(
        config.datasets.get("hf_dataset_name", "") or "parquet",
        data_files=data_files,
        revision=config.datasets.get("hf_revision", None),
        streaming=True,
        **load_kwargs,
    )

    train_dataset = _apply_streaming_filters(
        dataset["train"], config, "training", row_filter
    )
    val_dataset = None
    if val_num_shards > 0:
        val_dataset = _apply_streaming_filters(
            dataset["validation"], config, "validation", row_filter
        ).take(val_num_samples)

    return _finalize_streaming_splits(train_dataset, val_dataset, config)


def build_dataset(config: dict) -> Tuple[DataLoader, DataLoader]:
    if config.datasets.type == "hf_streaming_codes":
        return _build_streaming_codes_dataset(config)

    # Streaming Parquet path: keep the Hugging Face IterableDataset lazy and
    # avoid materializing metadata/audio in RAM.
    if config.datasets.type == "hf_streaming_text_tokenizer":
        val_metadata = config.datasets.get("val_metadata", "")

        data_files = {
            "train": config.datasets.train_metadata,
        }
        if val_metadata:
            data_files["validation"] = val_metadata

        load_kwargs, row_filter = _streaming_load_kwargs(config)

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

        return _finalize_streaming_splits(train_dataset, val_dataset, config)

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


    if config.datasets.type == "hf_text_tokenizer":
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