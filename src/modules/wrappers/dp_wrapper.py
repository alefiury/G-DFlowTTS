import os
import sys
from typing import Optional

sys.path.append(os.getcwd())

from typing import Tuple

import torch
import pandas as pd
from torch import Tensor
import lightning as L
from omegaconf import DictConfig
from torch.optim import AdamW
from transformers import AutoTokenizer
from lightning.pytorch.utilities import grad_norm

from modules.duration_predictor.model import DurationPredictor
from utils.lr_schedulers import CosineWarmupLR
from dataset.dataloader import (
    DurationBPEOfflineDataset,
    DurationBPEOfflineCollateFunc
)


class DurationPredictorWrapper(L.LightningModule):
    def __init__(
        self,
        config: DictConfig,
    ):
        super().__init__()
        self.config = config
        self.criteria = torch.nn.CrossEntropyLoss(reduction="none")
        self.model = DurationPredictor(**self.config.model)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            train_df = pd.read_csv(self.config.datasets.train_metadata)
            val_df = pd.read_csv(self.config.datasets.val_metadata)

            text_tokenizer = AutoTokenizer.from_pretrained(self.config.datasets.text_tokenizer_name)

            self.train_dataset = DurationBPEOfflineDataset(
                data=train_df,
                base_dir=self.config.datasets.base_dir,
                filepath_column=self.config.datasets.filepath_column,
                text_tokenizer=text_tokenizer,
            )
            self.val_dataset = DurationBPEOfflineDataset(
                data=val_df,
                base_dir=self.config.datasets.base_dir,
                filepath_column=self.config.datasets.filepath_column,
                text_tokenizer=text_tokenizer,
            )

    def train_dataloader(self):
        collate_fn = DurationBPEOfflineCollateFunc(
            audio_bos_token=self.config.datasets.audio_bos_token,
            audio_pad_token=self.config.datasets.audio_pad_token,
            text_pad_token=self.config.datasets.text_pad_token,
            max_audio_length=self.config.datasets.max_audio_length,
        )

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=self.config.train.shuffle,
            num_workers=self.config.train.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        collate_fn = DurationBPEOfflineCollateFunc(
            audio_bos_token=self.config.datasets.audio_bos_token,
            audio_pad_token=self.config.datasets.audio_pad_token,
            text_pad_token=self.config.datasets.text_pad_token,
            max_audio_length=self.config.datasets.max_audio_length,
        )

        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=False,
            num_workers=self.config.train.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
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

    def forward(
        self,
        text_ids: Tensor,
        audio_ids: Tensor,
        text_mask: Optional[Tensor] = None,
        audio_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        return self.model(
            text_ids=text_ids,
            text_mask=text_mask,
            audio_ids=audio_ids,
            audio_mask=audio_mask,
        )

    def _step(self, batch, batch_idx, split: str):
        audio, a_mask, _, text, t_mask, rem_len = batch  # unpack collate output

        logits = self(text_ids=text, audio_ids=audio, text_mask=t_mask, audio_mask=a_mask)
        # logits : [B, Tmax, V]
        B, Tmax, V = logits.shape

        # Flatten -----------------------------------------------------------
        logits_flat   = logits.reshape(-1, V)              # [(B*Tmax), V]
        targets_flat  = rem_len.reshape(-1)                # [(B*Tmax)]
        mask_flat     = a_mask.reshape(-1)                 # [(B*Tmax)] bool

        # print(Tmax, V)
        # print(torch.max(rem_len), torch.min(rem_len))
        # print(rem_len)

        # Cross‑entropy per position (no reduction) then mask
        loss_all = self.criteria(logits_flat, targets_flat)  # [B*Tmax]
        loss = (loss_all * mask_flat.float()).sum() / mask_flat.sum()

        self.log(f"{split}/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        # return loss
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")
