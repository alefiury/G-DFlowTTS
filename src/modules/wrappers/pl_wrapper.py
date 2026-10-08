import os
import sys
import random
from typing import Optional

sys.path.append(os.getcwd())

from typing import Tuple

import torch
import wandb
from tqdm import tqdm
import torchaudio
from torch import Tensor
import lightning as L
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.optim import AdamW
from transformers import AutoTokenizer
from lightning.pytorch.utilities import grad_norm

from torch.nn.modules.loss import _Loss
from flow_matching.path import MixtureDiscreteProbPath, ProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.loss import MixturePathGeneralizedKL

from modules.gdflowtts.model import Transformer
from dataset.build_dataset import build_dataset
from utils.lr_schedulers import CosineWarmupLR
from utils.neucodec import NeuCodec
from utils.inference_utils import sample_with_official_solver
from modules.gdflowtts.flow import MaskedSourceDistribution, UniformSourceDistribution

from dataset.dataloader import (
    HFTextTokenizerCollator,
    StreamingHFTextTokenizerCollator,
    StreamingHFCodesCollator,
)


# Dataset types streamed with Hugging Face IterableDatasets
STREAMING_DATASET_TYPES = ("hf_streaming_text_tokenizer", "hf_streaming_codes")
# Dataset types whose text is tokenized with a Hugging Face AutoTokenizer
HF_TOKENIZER_DATASET_TYPES = ("hf_text_tokenizer",) + STREAMING_DATASET_TYPES


def get_loss_function(loss_function: str, path: Optional[ProbPath] = None) -> _Loss:
    if loss_function == "cross_entropy":
        return torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    elif loss_function == "generalized_kl":
        assert path is not None, "Path must be provided for generalized_kl loss."
        return MixturePathGeneralizedKL(path=path, reduction="none")
    else:
        raise ValueError(f"{loss_function} is not supported")


class DFMTTSWrapper(L.LightningModule):
    def __init__(
        self,
        config: DictConfig,
    ):
        super().__init__()
        self.config = config

        if self.config.source_dist_type == "uniform":
            print("\n\nUsing Uniform Source Distribution!\n\n")
            self.source_distribution = UniformSourceDistribution(
                vocab_size=self.config.datasets.audio_vocab_size + self.config.model.audio_add_token
            )
        elif self.config.source_dist_type == "mask":
            print("\n\nUsing Masked Source Distribution!\n\n")
            self.source_distribution = MaskedSourceDistribution(
                mask_token=self.config.datasets.audio_mask_token,
                vocab_size=self.config.datasets.audio_vocab_size + self.config.model.audio_add_token,
            )
        else:
            raise ValueError(f"Invalid source distribution: {self.config.source_dist_type}")

        if self.config.get("scheduler_type", "polynomial") == "polynomial":
            self.path = MixtureDiscreteProbPath(
                scheduler=PolynomialConvexScheduler(n=1.0)
            )
        else:
            raise ValueError(f"Invalid scheduler type: {self.config.scheduler.type}")

        try:
            loss_function_name = self.config.loss.get("function", "cross_entropy")
        except:
            loss_function_name = "cross_entropy"
        self.criteria = get_loss_function(loss_function=loss_function_name, path=self.path)

        if self.config.model_type.lower() == "dit_adaln":
            self.model = Transformer(**self.config.model)
        else:
            raise ValueError(f"Invalid model type: {self.config.model_type}")

        if config.datasets.type == "hf_streaming_text_tokenizer":
            if config.datasets.get("codec_name", "") == "neucodec":
                # Do not call .cuda() here; Lightning owns device placement.
                self.audio_codec = NeuCodec.from_pretrained("neuphonic/neucodec")
                self.audio_codec.eval()
                self.audio_codec.requires_grad_(False)
            else:
                raise ValueError(f"Invalid codec name: {config.datasets.codec_name}")

    def on_save_checkpoint(self, checkpoint):
        # Remove all parameters whose keys start with "audio_codec"
        state_dict = checkpoint["state_dict"]
        keys_to_remove = [key for key in state_dict if key.startswith("audio_codec")]
        for key in keys_to_remove:
            del state_dict[key]

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_dataset, self.val_dataset = build_dataset(self.config)

            # Iterable datasets do not get a DistributedSampler. Split the
            # streamed shards/examples explicitly across DDP ranks.
            if (
                self.config.datasets.type in STREAMING_DATASET_TYPES
                and self.trainer.world_size > 1
            ):
                from datasets.distributed import split_dataset_by_node

                self.train_dataset = split_dataset_by_node(
                    self.train_dataset,
                    rank=self.global_rank,
                    world_size=self.trainer.world_size,
                )
                if self.val_dataset is not None:
                    self.val_dataset = split_dataset_by_node(
                        self.val_dataset,
                        rank=self.global_rank,
                        world_size=self.trainer.world_size,
                    )

    def on_train_epoch_start(self):
        # Streamed datasets reshuffle shards and buffer from seed + epoch.
        if hasattr(self.train_dataset, "set_epoch"):
            self.train_dataset.set_epoch(self.current_epoch)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Keep raw streamed audio on CPU for NeuCodec preprocessing.

        NeuCodec's feature extractor converts the raw waveform to NumPy/CPU
        before moving extracted features to the codec device.  Moving the
        waveform to CUDA here would cause an unnecessary CPU -> GPU -> CPU
        round trip.
        """
        if self.config.datasets.type == "hf_streaming_text_tokenizer":
            if batch is None:
                # Every sample in the batch failed to decode; skip it.
                return None
            waveforms, transcription_ids, transcription_att_mask = batch
            return (
                waveforms,
                transcription_ids.to(device, non_blocking=True),
                transcription_att_mask.to(device, non_blocking=True),
            )

        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def _build_collate_fn(self):
        """Return the collate function for the configured dataset type."""
        datasets_config = self.config.datasets
        if datasets_config.type in HF_TOKENIZER_DATASET_TYPES:
            text_tokenizer = AutoTokenizer.from_pretrained(datasets_config.text_tokenizer_name)
            if text_tokenizer.pad_token is None:
                text_tokenizer.add_special_tokens({"pad_token": text_tokenizer.eos_token})

            if datasets_config.type == "hf_streaming_text_tokenizer":
                return StreamingHFTextTokenizerCollator(
                    text_tokenizer=text_tokenizer,
                    text_column=datasets_config.text_column,
                    audio_column=datasets_config.get("audio_column", "audio"),
                    sampling_rate=datasets_config.sampling_rate,
                    max_audio_duration=datasets_config.max_audio_duration,
                    id_column=datasets_config.get("filepath_column", "filepath"),
                )
            codes_collator = HFTextTokenizerCollator(
                text_tokenizer=text_tokenizer,
                max_audio_length=datasets_config.max_audio_length,
                audio_pad_token=getattr(datasets_config, "audio_pad_token", None),
                audio_eos_token=getattr(datasets_config, "audio_eos_token", None),
                audio_pad_type=datasets_config.audio_pad_type,
                use_eos_as_pad=datasets_config.use_eos_as_pad,
            )
            if datasets_config.type == "hf_streaming_codes":
                return StreamingHFCodesCollator(
                    codes_collator=codes_collator,
                    text_column=datasets_config.get("text_column", "text"),
                    codes_column=datasets_config.get("codes_column", "codes"),
                    id_column=datasets_config.get("id_column", "_id"),
                )
            return codes_collator
        raise ValueError(f"Invalid dataset type: {datasets_config.type}")

    def train_dataloader(self):
        """Return the training dataloader."""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=(
                False
                if self.config.datasets.type in STREAMING_DATASET_TYPES
                else self.config.train.shuffle
            ),
            num_workers=self.config.train.num_workers,
            pin_memory=True,
            collate_fn=self._build_collate_fn(),
        )

    def val_dataloader(self):
        if getattr(self, "val_dataset", None) is None:
            return None

        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=False,
            num_workers=self.config.train.num_workers,
            pin_memory=True,
            collate_fn=self._build_collate_fn(),
        )

    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = self.train_dataloader()
        if self.trainer.max_steps and self.trainer.max_steps > 0:
            return self.trainer.max_steps
        dataset_size = len(dataset)

        gpu_count = self.trainer.num_devices if self.trainer.num_devices else 1
        accumulate_grad_batches = self.trainer.accumulate_grad_batches

        effective_batches = dataset_size // (gpu_count * accumulate_grad_batches)

        return effective_batches * self.trainer.max_epochs

    def configure_optimizers(self):
        """Configures the optimizer and the learning rate scheduler."""
        # Start dataloaders to be able to get the number of steps per epoch
        self.trainer.fit_loop.setup_data()

        max_num_steps = self.num_training_steps()

        print(f"Max number of steps: {max_num_steps}")

        opt_params = self.config.optimizer["params"]
        scheduler_params = self.config.scheduler["params"]

        # Exclude bias and normalization layers from weight decay
        # As it can be seen in https://arxiv.org/pdf/2106.15739 and https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994
        # Inspired by https://github.com/mlfoundations/open_clip/blob/49eac2f27a5bb98a7f7ecc1154918880aa55256c/src/open_clip_train/main.py#L312
        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(self.model.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

        if self.config.optimizer.name.lower() != "adamw":
            raise ValueError(f"Invalid optimizer: {self.config.optimizer.name}")
        optimizer = AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": opt_params["weight_decay"]},
            ],
            lr=opt_params["learning_rate"],
            eps=opt_params["eps"],
            betas=opt_params["betas"],
            weight_decay=opt_params["weight_decay"]
        )

        if not self.config["scheduler"]:
            return optimizer

        if self.config.scheduler.name.lower() != "cosinewarmuplr":
            raise ValueError(f"Invalid scheduler: {self.config.scheduler.name}")
        scheduler = CosineWarmupLR(
            optimizer,
            lr_min=opt_params.get("min_learning_rate", 1.0e-6),
            lr_max=opt_params["learning_rate"],
            warmup=scheduler_params.get("warmup_lr", max_num_steps*0.05),
            T_max=max_num_steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)

    @torch.no_grad()
    def encode_streaming_waveforms(self, waveforms):
        """Encode variable-length raw utterances and reproduce offline padding."""
        if self.config.datasets.codec_name != "neucodec":
            raise NotImplementedError(
                "Raw Parquet streaming is currently implemented for NeuCodec."
            )

        # Lightning calls train() recursively on submodules; force the frozen
        # codec back to eval mode before extracting discrete targets.
        self.audio_codec.eval()

        code_list = []
        for waveform in waveforms:
            # Keep raw audio on CPU: the NeuCodec feature extractor computes
            # the mel features there, then the codec moves them to its device.
            waveform = waveform.detach().to(device="cpu", dtype=torch.float32)
            if waveform.ndim == 2:
                waveform = waveform.unsqueeze(0)  # [1, 1, T]
            if waveform.ndim != 3:
                raise ValueError(
                    f"NeuCodec input must be [B,1,T], got {tuple(waveform.shape)}"
                )

            codes = self.audio_codec.encode_code(waveform).squeeze().long()
            if codes.ndim != 1:
                raise ValueError(
                    f"Expected 1-D NeuCodec sequence, got {tuple(codes.shape)}"
                )
            code_list.append(codes)

        if self.config.datasets.audio_pad_type == "variable":
            max_audio_length = max(code.numel() + 1 for code in code_list)
            max_audio_length = min(
                max_audio_length,
                self.config.datasets.max_audio_length,
            )
        elif self.config.datasets.audio_pad_type == "fixed":
            max_audio_length = self.config.datasets.max_audio_length
        else:
            raise ValueError(
                f"Unknown audio_pad_type: {self.config.datasets.audio_pad_type}"
            )

        effective_pad_id = (
            self.config.datasets.audio_eos_token
            if self.config.datasets.use_eos_as_pad
            else self.config.datasets.audio_pad_token
        )
        if effective_pad_id is None:
            raise ValueError(
                "audio_pad_token must be set when use_eos_as_pad is false"
            )

        padded_codes = []
        lengths_with_eos = []
        for codes in code_list:
            # Reserve one slot for the gold EOS token.
            codes = codes[: max_audio_length - 1]
            sequence = torch.cat(
                [
                    codes,
                    torch.tensor(
                        [self.config.datasets.audio_eos_token],
                        dtype=codes.dtype,
                        device=codes.device,
                    ),
                ]
            )
            lengths_with_eos.append(sequence.numel())

            if sequence.numel() < max_audio_length:
                sequence = F.pad(
                    sequence,
                    (0, max_audio_length - sequence.numel()),
                    value=effective_pad_id,
                )
            padded_codes.append(sequence)

        x_1 = torch.stack(padded_codes, dim=0)
        lengths = torch.tensor(
            lengths_with_eos,
            dtype=torch.long,
            device=x_1.device,
        )
        positions = torch.arange(x_1.size(1), device=x_1.device).unsqueeze(0)
        audio_att_mask = positions < lengths.unsqueeze(1)

        return x_1, audio_att_mask

    def build_keep_prefix(
        self,
        x1: torch.Tensor,                 # [B, L]
        audio_att_mask: torch.Tensor,      # [B, L] bool
        fps: int = 50,
        min_sec: float = 0.0,
        max_sec: float = 15.0,
        ensure_min_gen_sec: float = 0.5,  # keep at least this much to generate
    ) -> torch.Tensor:
        B, L = x1.shape
        device = x1.device

        valid_len = audio_att_mask.long().sum(dim=1)  # [B]
        minP = torch.round(torch.tensor(min_sec * fps, device=device)).long()
        maxP = torch.round(torch.tensor(max_sec * fps, device=device)).long()
        minGen = torch.round(torch.tensor(ensure_min_gen_sec * fps, device=device)).long()

        # For each sample, cap maxP so we still have something to generate
        maxP_i = torch.clamp(valid_len - minGen, min=minP, max=maxP)

        # Sample P per sample in [minP, maxP_i]
        # (torch.randint high is exclusive)
        P = torch.empty((B,), device=device, dtype=torch.long)
        for b in range(B):
            hi = int(maxP_i[b].item())
            lo = int(minP.item())
            if hi <= lo:
                P[b] = lo
            else:
                P[b] = torch.randint(lo, hi + 1, (1,), device=device)

        idx = torch.arange(L, device=device).unsqueeze(0)  # [1,L]
        keep = idx < P.unsqueeze(1)                        # [B,L]
        keep = keep & audio_att_mask.bool()                # never keep padding
        return keep

    def forward(
        self,
        x_t: Tensor,
        text_ids: Tensor,
        time: Tensor,
        drop_text: bool = False,
        text_att_mask: Tensor = None,
        audio_att_mask: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        return self.model(
            x_t=x_t,
            text=text_ids,
            time=time,
            drop_text=drop_text,
            text_att_mask=text_att_mask,
            audio_att_mask=audio_att_mask,
        )

    def training_step(self, batch, batch_idx):
        if batch is None:
            # Streaming collator dropped every sample in the batch; skip step.
            return None

        if self.config.datasets.type == "hf_streaming_text_tokenizer":
            input_waveforms, transcription_ids, transcription_att_mask = batch
            x_1, audio_att_mask = self.encode_streaming_waveforms(input_waveforms)
            x_0 = self.source_distribution.sample(x_1.shape, device=x_1.device)
        else:
            if len(batch) == 6:
                x_1, transcription_ids, transcription_att_mask, x_0, audio_att_mask, loss_weight_extra = batch
            elif len(batch) == 5:
                x_1, transcription_ids, transcription_att_mask, x_0, audio_att_mask = batch
                loss_weight_extra = None
            elif len(batch) == 4:
                x_1, audio_att_mask, transcription_ids, transcription_att_mask = batch
                x_0 = self.source_distribution.sample(x_1.shape, device=x_1.device)
            else:
                raise ValueError(f"Invalid number of elements in batch: {len(batch)}")

        with torch.no_grad():
            keep_prefix = None
            if self.config.datasets.get("use_prefix_mask", False):
                keep_prefix = self.build_keep_prefix(
                    x1=x_1,
                    audio_att_mask=audio_att_mask,
                    fps=self.config.datasets.get("fps", 50),
                    min_sec=self.config.datasets.get("prefix_min_sec", 0.0),
                    max_sec=self.config.datasets.get("prefix_max_sec", 15.0),
                    ensure_min_gen_sec=self.config.datasets.get("prefix_ensure_min_gen_sec", 0.5),
                )
                x_0 = torch.where(keep_prefix, x_1, x_0)

            # Elbo may have singularity at 1
            time_epsilon = 1e-3 if isinstance(self.criteria, MixturePathGeneralizedKL) else 0.0
            t = torch.rand(x_1.shape[0], device=x_1.device) * (1.0 - time_epsilon)
            path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)

        if random.random() < self.config.datasets.cond_drop_prob:
            drop_text = True
        else:
            drop_text = False

        # Assert to verify correctness when the conditional drop probability is set to 0
        if self.config.datasets.assert_drop_prob and self.config.datasets.cond_drop_prob == 0:
            assert drop_text == False , "Drop text should be False for training when cond_drop_prob is 0"

        # print(f"TRAINER - TRAIN: {times_t}")
        logits = self(
            x_t=path_sample.x_t,
            text_ids=transcription_ids,
            time=path_sample.t,
            drop_text=drop_text,
            audio_att_mask=audio_att_mask,
            text_att_mask=transcription_att_mask,
        ).float()

        if self.config.datasets.get("use_prefix_mask", False):
            audio_att_mask = audio_att_mask & (~keep_prefix)

        if self.config.loss.function == "generalized_kl":
            loss = self.criteria(
                logits=logits,
                x_1=x_1.long(),
                x_t=path_sample.x_t.long(),
                t=path_sample.t,
            )
            mask = audio_att_mask
            loss = (loss * mask).sum() / (mask.sum().clamp_min(1))

            with torch.no_grad():
                # use audio mask to transform padding positions to -100 so that they are ignored in the CE loss
                ce_target = x_1.masked_fill(~audio_att_mask.bool(), -100)
                loss_per = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    ce_target.view(-1).long(),
                    ignore_index=-100,
                    reduction="none",
                ).view_as(x_1)
                mask = audio_att_mask.bool()
                aux_ce_loss = (loss_per * mask).sum() / mask.sum().clamp_min(1)
            self.log("train/aux_ce_loss", aux_ce_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        else:
            # use audio mask to transform padding positions to -100 so that they are ignored in the CE loss
            target = x_1.masked_fill(~audio_att_mask.bool(), -100)

            loss_per = self.criteria(
                input=logits.view(-1, logits.size(-1)),
                target=target.view(-1).long()
            ).view_as(x_1)  # [B, L]

            mask = audio_att_mask.bool()
            loss = (loss_per * mask).sum() / mask.sum().clamp_min(1)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if batch is None:
            # Streaming collator dropped every sample in the batch; skip step.
            return None

        if self.config.datasets.type == "hf_streaming_text_tokenizer":
            input_waveforms, transcription_ids, transcription_att_mask = batch
            x_1, audio_att_mask = self.encode_streaming_waveforms(input_waveforms)
            x_0 = self.source_distribution.sample(x_1.shape, device=x_1.device)
        else:
            if len(batch) == 6:
                x_1, transcription_ids, transcription_att_mask, x_0, audio_att_mask, loss_weight_extra = batch
            elif len(batch) == 5:
                x_1, transcription_ids, transcription_att_mask, x_0, audio_att_mask = batch
                loss_weight_extra = None
            elif len(batch) == 4:
                x_1, audio_att_mask, transcription_ids, transcription_att_mask = batch
                x_0 = self.source_distribution.sample(x_1.shape, device=x_1.device)
            else:
                raise ValueError(f"Invalid number of elements in batch: {len(batch)}")

        with torch.no_grad():
            if self.config.datasets.get("use_prefix_mask", False):
                keep_prefix = self.build_keep_prefix(
                    x1=x_1,
                    audio_att_mask=audio_att_mask,
                    fps=self.config.datasets.get("fps", 50),
                    min_sec=self.config.datasets.get("prefix_min_sec", 0.0),
                    max_sec=self.config.datasets.get("prefix_max_sec", 15.0),
                    ensure_min_gen_sec=self.config.datasets.get("prefix_ensure_min_gen_sec", 0.5),
                )
                x_0 = torch.where(keep_prefix, x_1, x_0)

            time_epsilon = 1e-3 if isinstance(self.criteria, MixturePathGeneralizedKL) else 0.0
            t = torch.rand(x_1.shape[0], device=x_1.device) * (1.0 - time_epsilon)
            path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)

        # print(f"TRAINER - VAL: {times_t}")
        logits = self(
            x_t=path_sample.x_t,
            text_ids=transcription_ids,
            time=path_sample.t,
            drop_text=False,
            audio_att_mask=audio_att_mask,
            text_att_mask=transcription_att_mask,
        ).float()

        if self.config.datasets.get("use_prefix_mask", False):
            audio_att_mask = audio_att_mask & (~keep_prefix)

        if self.config.loss.function == "generalized_kl":
            loss = self.criteria(
                logits=logits,
                x_1=x_1.long(),
                x_t=path_sample.x_t.long(),
                t=path_sample.t,
            )
            mask = audio_att_mask
            loss = (loss * mask).sum() / (mask.sum().clamp_min(1))

            with torch.no_grad():
                ce_target = x_1.masked_fill(~audio_att_mask.bool(), -100)

                loss_per = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    ce_target.view(-1).long(),
                    ignore_index=-100,
                    reduction="none",
                ).view_as(x_1)
                mask = audio_att_mask.bool()
                aux_ce_loss = (loss_per * mask).sum() / mask.sum().clamp_min(1)
            self.log("val/aux_ce_loss", aux_ce_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        else:
            # use audio mask to transform padding positions to -100 so that they are ignored in the CE loss
            target = x_1.masked_fill(~audio_att_mask.bool(), -100)

            loss_per = self.criteria(
                input=logits.view(-1, logits.size(-1)),
                target=target.view(-1).long()
            ).view_as(x_1)  # [B, L]

            mask = audio_att_mask.bool()
            loss = (loss_per * mask).sum() / mask.sum().clamp_min(1)

        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if batch_idx == 0 and self.config.test.get("log_audio_ref", False):
            try:
                self.sample_validation()
            except Exception as e:
                print(f"Error during validation sample generation: {e}")
                wandb.log({"validation_sample": None})
        return loss

    @torch.no_grad()
    def sample_validation(self):
        audio_ref_path = self.config.test.audio_ref_path
        text_ref = self.config.test.text_ref

        if self.config.datasets.codec_name != "neucodec":
            raise ValueError(f"Invalid codec name: {self.config.datasets.codec_name}")
        audio_codec = NeuCodec.from_pretrained("neuphonic/neucodec").to(self.device)
        audio_codec.eval()
        saving_sr = 24000

        audio_ref, audio_ref_sr = torchaudio.load(audio_ref_path)
        if audio_ref_sr != self.config.datasets.sampling_rate:
            audio_ref = torchaudio.transforms.Resample(audio_ref_sr, self.config.datasets.sampling_rate)(audio_ref)

        codes_ref = audio_codec.encode_code(audio_ref[None, ...]).squeeze()
        codes_ref_size = codes_ref.shape[-1]
        codes_ref = codes_ref.unsqueeze(0).to(self.device)

        if self.config.datasets.type not in HF_TOKENIZER_DATASET_TYPES:
            raise ValueError(f"Invalid text tokenizer type: {self.config.datasets.type}")
        text_tokenizer = AutoTokenizer.from_pretrained(self.config.datasets.text_tokenizer_name)
        if text_tokenizer.pad_token is None:
            text_tokenizer.add_special_tokens({"pad_token": text_tokenizer.eos_token})

        max_length = self.config.test.max_audio_length
        generated_audios = {}
        # Iterate over each test sentence from config
        for idx, sentence in tqdm(enumerate(self.config.test.sentences), total=len(self.config.test.sentences)):
            # if text_ref has a final point, remove it for better concatenation
            if text_ref.endswith("."):
                text_ref = text_ref[:-1]
            augmented_sentence = text_ref + ". " + sentence

            text_ids = text_tokenizer(
                augmented_sentence,
                return_tensors="pt"
            )["input_ids"].squeeze(0).to(self.device).unsqueeze(0)

            # Initialize xt from the source distribution (batch size = 1)
            x_t = self.source_distribution.sample((1, max_length), device=self.device)

            if self.config.datasets.cond_drop_prob==0.0:
                x_t = sample_with_official_solver(
                    config=self.config,
                    model=self,
                    text_ids=text_ids,
                    text_att_mask=None,
                    codes_ref_1d=codes_ref.squeeze(0),
                    suffix_len=codes_ref_size,
                    steps=self.config.test.nsf,
                    device=self.device,
                )
            elif self.config.source_dist_type == "mask":
                x_t = self.generate_sample_pfg_mask(
                    xt=x_t,
                    text_ids=text_ids,
                    codes_ref=codes_ref,
                    nsf=self.config.test.nsf,
                    codes_ref_size=codes_ref_size
                )
            else:
                x_t = self.generate_sample_pfg_euler_uniform(
                    xt=x_t,
                    text_ids=text_ids,
                    codes_ref=codes_ref,
                    nsf=self.config.test.nsf,
                    codes_ref_size=codes_ref_size
                )

            # Drop special tokens (>= codec vocab), they would crash decode_code
            codec_vocab = int(self.config.datasets.audio_vocab_size)
            if int(x_t.max()) >= codec_vocab:
                x_t = x_t[x_t < codec_vocab]
            # if x_t is not in the shape (1, 1, T), reshape it
            if x_t.dim() == 2:
                x_t = x_t.unsqueeze(0)
            elif x_t.dim() == 1:
                x_t = x_t.unsqueeze(0).unsqueeze(0)
            generated_audio = audio_codec.decode_code(x_t)
            generated_audios[f"generated_audio_{idx}"] = wandb.Audio(
                generated_audio[0, 0, :].cpu().numpy(),
                sample_rate=saving_sr,
                caption=sentence
            )
        wandb.log(generated_audios)

    def generate_sample_pfg_mask(self, xt, text_ids, codes_ref, nsf: int, codes_ref_size: int):
        num_steps = nsf
        dt = 1.0 / num_steps
        x1_temp = 1.0
        gamma = 2.5
        mask_token_id = self.config.datasets.audio_mask_token
        S = self.config.datasets.audio_vocab_size + self.config.model.audio_add_token
        eps = 1e-9
        noise = 0.0

        mask_one_hot = torch.zeros((S), device=self.device)
        mask_one_hot[mask_token_id] = 1.0

        xt[..., : codes_ref_size] = codes_ref[..., : codes_ref_size]

        # create text att_mask, all elements are "true" because we only have one sample
        text_att_mask = text_ids.new_ones((1, text_ids.size(1)), dtype=torch.bool)

        # Loop over the time grid
        for step in tqdm(range(num_steps), total=num_steps):
            t_val    = step * dt
            t_tensor = xt.new_full((1,), t_val, dtype=torch.float32, device=self.device)

            # unconditional pass
            logits_u = self(
                x_t=xt,
                text_ids=text_ids,
                text_att_mask=text_att_mask,
                time=t_tensor,
                drop_text=True,
            )
            probs_u  = torch.softmax(logits_u / x1_temp, -1)

            # conditional pass
            logits_c = self(
                x_t=xt,
                text_ids=text_ids,
                text_att_mask=text_att_mask,
                time=t_tensor,
                drop_text=False
            )
            probs_c  = torch.softmax(logits_c / x1_temp, -1)

            xt_mask  = (xt == mask_token_id).unsqueeze(-1).float()
            base_r   = (1 + noise * t_val) / (1 - t_val)

            R_u = xt_mask * probs_u * base_r
            R_c = xt_mask * probs_c * base_r

            remask = (1 - xt_mask) * mask_one_hot.view(1,1,S) * noise
            R_u += remask;  R_c += remask

            log_Ru = torch.log(R_u + eps)
            log_Rc = torch.log(R_c + eps)
            R_mix  = torch.exp(gamma * log_Rc + (1 - gamma) * log_Ru)

            # enforce row‑sum zero
            R_mix.scatter_(-1, xt[..., None], 0.)
            R_mix.scatter_(-1, xt[..., None], -R_mix.sum(-1, keepdim=True))

            # Euler step
            P = (R_mix * dt).clamp_min(0.)
            diag = (1. - P.sum(-1, keepdim=True)).clamp_min(0.)
            P.scatter_(-1, xt[..., None], diag)

            xt = torch.multinomial(P.view(-1, S), 1).view_as(xt)

            # Force conditioning
            xt[..., : codes_ref_size] = codes_ref[..., : codes_ref_size]

        # Drop the prompt, then remove EOS, mask and padding tokens
        xt = xt[..., codes_ref_size:].squeeze(0)
        xt = xt[xt != self.config.datasets.audio_eos_token]
        xt = xt[xt != self.config.datasets.audio_mask_token]
        if hasattr(self.config.datasets, "audio_pad_token"):
            xt = xt[xt != self.config.datasets.audio_pad_token]

        return xt.unsqueeze(0).unsqueeze(0)

    @torch.no_grad()
    def generate_sample_pfg_euler_uniform(
        self,
        xt: torch.Tensor,                 # [B, T]
        text_ids: torch.Tensor,           # [B, L]
        codes_ref: torch.Tensor,          # [B, T] or [B, prefix] (you use [:codes_ref_size])
        nsf: int,
        codes_ref_size: int,
        *,
        x1_temp: float = 1.0,
        guidance_scale: float = 2.5,      # CFG/PFG scale; 1.0 == conditional only
        eps: float = 1e-12,
        clamp_logp: float = 80.0,         # numerical safety before exp/softmax
    ) -> torch.Tensor:
        """
        Official MixtureDiscreteEuler-style CTMC step, but using Predictor-Free Guidance (PFG)
        to form a guided posterior p_{1|t} before sampling x1.

        Assumptions:
        - div_free == 0
        - source distribution is uniform (so we don't need p0 / divergence-free term)
        - PFG only makes sense if training used cond_drop_prob > 0

        Returns:
            audio token ids shaped [B, 1, T_gen] (padded across batch)
        """
        B, T = xt.shape
        num_steps = int(nsf)
        dt = 1.0 / max(1, num_steps)

        S = int(self.config.datasets.audio_vocab_size + self.config.model.audio_add_token)

        # tokens
        eos_id  = int(getattr(self.config.datasets, "audio_eos_token", -1))
        pad_id  = getattr(self.config.datasets, "audio_pad_token", None)
        pad_id  = int(pad_id) if pad_id is not None else None

        # PFG sanity
        cond_drop_prob = float(getattr(self.config.datasets, "cond_drop_prob", 0.0))
        if guidance_scale != 1.0 and cond_drop_prob <= 0.0:
            raise RuntimeError(
                "guidance_scale != 1.0 but config.datasets.cond_drop_prob==0.0. "
                "This checkpoint likely never learned the unconditional (drop_text=True) branch."
            )

        # pin prefix
        xt = xt.clone()
        xt[..., :codes_ref_size] = codes_ref[..., :codes_ref_size]

        # audio attention mask if your model uses it
        audio_att_mask = torch.ones_like(xt, dtype=torch.bool)

        for step in range(num_steps):
            t_val = step * dt
            t = xt.new_full((B,), t_val, dtype=torch.float32)

            # ---- model posteriors p_{1|t} ----
            logits_u = self(
                x_t=xt,
                text_ids=text_ids,
                audio_att_mask=audio_att_mask,
                time=t,
                drop_text=True,
            ).float()
            logits_c = self(
                x_t=xt,
                text_ids=text_ids,
                audio_att_mask=audio_att_mask,
                time=t,
                drop_text=False,
            ).float()

            temp = max(1e-3, float(x1_temp))

            # Work in log-prob space for stable PFG:
            logp_u = F.log_softmax(logits_u / temp, dim=-1).clamp(-clamp_logp, clamp_logp)
            logp_c = F.log_softmax(logits_c / temp, dim=-1).clamp(-clamp_logp, clamp_logp)

            s = float(guidance_scale)
            # CFG/PFG mixing: log p = log p_u + s * (log p_c - log p_u)
            logp = logp_u + s * (logp_c - logp_u)

            # p_{1|t}
            p1t = torch.softmax(logp, dim=-1)  # [B, T, S]
            p1t = torch.nan_to_num(p1t, nan=0.0, posinf=0.0, neginf=0.0)
            p1t = p1t / p1t.sum(dim=-1, keepdim=True).clamp_min(eps)

            # sample x1 ~ p_{1|t}(\cdot|x_t)
            x1 = torch.multinomial(p1t.view(-1, S), 1).view(B, T)

            # final step in official solver: directly set x_t = x1
            if step == num_steps - 1:
                xt = x1
                xt[..., :codes_ref_size] = codes_ref[..., :codes_ref_size]
                break

            # ---- official Euler CTMC rates u = lambda(t) * delta_{x1} (no div_free) ----
            sched = self.path.scheduler(t)  # MixtureDiscreteProbPath scheduler
            alpha_t = sched.alpha_t
            dalpha_t = sched.d_alpha_t.clamp_min(1e-6)

            lam = (dalpha_t / (1.0 - alpha_t).clamp_min(1e-6)).view(B, 1, 1)  # [B,1,1]

            delta_1 = F.one_hot(x1, num_classes=S).to(lam.dtype)               # [B,T,S]
            u = lam * delta_1                                                  # [B,T,S]

            # Set u(x_t | x_t, x1) = 0 (remove diagonal/current state)
            delta_t = F.one_hot(xt, num_classes=S).to(torch.bool)              # [B,T,S]
            u = torch.where(delta_t, torch.zeros_like(u), u)

            # hazard/intensity per position
            hazard = u.sum(dim=-1)                                             # [B,T]

            # jump probability tau-leap
            p_jump = 1.0 - torch.exp(-dt * hazard)
            do_jump = (torch.rand_like(p_jump) < p_jump)

            # never change prefix
            do_jump[..., :codes_ref_size] = False

            # sample jump destination from u/hazard (only where do_jump)
            if do_jump.any():
                hazard_safe = hazard.clamp_min(eps)
                q = u / hazard_safe.unsqueeze(-1)                              # [B,T,S]
                q2 = q.view(-1, S)
                jump_idx = do_jump.view(-1).nonzero(as_tuple=False).squeeze(-1)

                q_jump = q2.index_select(0, jump_idx)
                q_jump = torch.nan_to_num(q_jump, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

                # (should already be normalized, but be safe)
                q_jump = q_jump / q_jump.sum(dim=-1, keepdim=True).clamp_min(eps)

                sampled = torch.multinomial(q_jump, 1).squeeze(-1)

                xt_flat = xt.view(-1)
                xt_flat[jump_idx] = sampled.to(xt_flat.dtype)
                xt = xt_flat.view_as(xt)

            # re-pin prefix
            xt[..., :codes_ref_size] = codes_ref[..., :codes_ref_size]

        # ---- postprocess: remove prefix, truncate at first EOS, pad batch ----
        outs = []
        for b in range(B):
            seq = xt[b, codes_ref_size:].detach()

            # truncate at first EOS
            if eos_id is not None and eos_id >= 0:
                pos = (seq == eos_id).nonzero(as_tuple=False)
                if pos.numel() > 0:
                    seq = seq[: int(pos[0].item())]

            if pad_id is not None:
                seq = seq[seq != pad_id]

            outs.append(seq)

        max_len = max([int(x.numel()) for x in outs]) if outs else 0
        if max_len == 0:
            fill = pad_id if pad_id is not None else (eos_id if eos_id >= 0 else 0)
            return xt.new_full((B, 1, 1), int(fill), dtype=torch.long)

        fill = pad_id if pad_id is not None else (eos_id if eos_id >= 0 else 0)
        out = xt.new_full((B, max_len), int(fill), dtype=torch.long)
        for b, seq in enumerate(outs):
            L = int(seq.numel())
            if L > 0:
                out[b, :L] = seq

        return out.unsqueeze(1)  # [B,1,Tgen]