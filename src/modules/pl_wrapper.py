import os
import sys
import random
from dataclasses import dataclass, field
from typing import Optional

sys.path.append(os.getcwd())

from typing import Tuple, List, Dict, Any, Union

import torch
import wandb
from tqdm import tqdm
import torchaudio
from torch import Tensor
import lightning as L
from lion_pytorch import Lion
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.optim import Adam, AdamW
from transformers import AutoTokenizer
from lightning.pytorch.utilities import grad_norm
from xcodec2.modeling_xcodec2 import XCodec2Model
from neucodec import NeuCodec
from torch.distributions.categorical import Categorical

from torch.nn.modules.loss import _Loss
from flow_matching.path import MixtureDiscreteProbPath, ProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.loss import MixturePathGeneralizedKL

from modules.model import Transformer
from modules.model_cross_att import TransformerCrossAttn
from utils.tokenizer import VoiceBpeTokenizer
from dataset.build_dataset import build_dataset
from utils.lr_schedulers import CosineWarmupLR, LinearLR
from utils.phonemes_tokenizer import PhonemeTokenizer
from modules.flow import KOConvexScheduler, MaskedSourceDistribution, UniformSourceDistribution

from dataset.dataloader import (
    DynamicSingleSpeakerCollateFunc,
    OfflineMultipleSpeakerMaskCollateFunc,
    OfflineMultipleSpeakerDreamOnCollateFunc,
    OfflineVoiceCloningSimplifiedCollateFunc,
    HFTextTokenizerCollator,
    PhonemeTokenizerCollator,
)


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
                mask_token=self.config.datasets.audio_mask_token
            )
        else:
            raise ValueError(f"Invalid source distribution: {self.config.source_dist_type}")

        if self.config.get("scheduler_type", "polynomial") == "ko":
            print("\n\nUsing KO scheduler!\n\n")
            self.path = MixtureDiscreteProbPath(
                scheduler=KOConvexScheduler()
            )
        elif self.config.get("scheduler_type", "polynomial") == "polynomial":
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
        elif self.config.model_type.lower() == "dit_crossattn":
            self.model = TransformerCrossAttn(**self.config.model)
        else:
            raise ValueError(f"Invalid model type: {self.config.model_type}")

        if config.datasets.type == "dynamic":
            if config.datasets.get("codec_name", "") == "xcodec2":
                self.audio_codec = XCodec2Model.from_pretrained(self.config.datasets.audio_codec)
            elif config.datasets.get("codec_name", "") == "neucodec":
                model = NeuCodec.from_pretrained("neuphonic/neucodec")
                model.eval().cuda()
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

    def train_dataloader(self):
        """Return the training dataloader."""
        if self.config.datasets.type == "dynamic":
            collate_fn = DynamicSingleSpeakerCollateFunc()
        elif self.config.datasets.type == "offline":
            collate_fn = OfflineMultipleSpeakerMaskCollateFunc(
                max_audio_length=self.config.datasets.max_audio_length,
                mask_prob=self.config.datasets.mask_prob,
                audio_mask_token=self.config.datasets.audio_mask_token,
                audio_pad_token=getattr(self.config.datasets, "audio_pad_token", None),
                audio_eos_token=self.config.datasets.audio_eos_token,
                text_pad_token=self.config.datasets.text_pad_token,
                mask_type=self.config.datasets.mask_type,
                audio_pad_type=self.config.datasets.audio_pad_type,
                use_eos_as_pad=self.config.datasets.use_eos_as_pad,
                loss_on_eos_pad=self.config.datasets.loss_on_eos_pad,
                pad_loss_weight=getattr(self.config.datasets, "pad_loss_weight", 1.0),
            )
        elif self.config.datasets.type == "offline_dynamic_dur":
            print("Creating collate function for offline_dynamic_dur")
            collate_fn = OfflineMultipleSpeakerDreamOnCollateFunc(
                max_audio_length=self.config.datasets.max_audio_length,
                audio_mask_token=self.config.datasets.audio_mask_token,
                audio_pad_token=getattr(self.config.datasets, "audio_pad_token", None),
                audio_expand_token=getattr(self.config.datasets, "audio_expand_token", None),
                audio_eos_token=getattr(self.config.datasets, "audio_eos_token", None),
                text_pad_token=self.config.datasets.text_pad_token,
                mask_type=self.config.datasets.mask_type,
                mask_prob=self.config.datasets.mask_prob,
                audio_pad_type=self.config.datasets.audio_pad_type,
                # Dynamic Duration Params
                mix_ratio=self.config.datasets.mix_ratio,
                p_merge_static=self.config.datasets.p_merge_static,
                p_merge_dynamic_scale=self.config.datasets.p_merge_dynamic_scale,
                delete_frac_range=self.config.datasets.delete_frac_range,
                # padding params
                use_eos_as_pad=self.config.datasets.use_eos_as_pad,
                loss_on_eos_pad=self.config.datasets.loss_on_eos_pad,
                pad_loss_weight=getattr(self.config.datasets, "pad_loss_weight", 1.0),
            )
        elif self.config.datasets.type == "offline_voice_cloning_simplified":
            collate_fn = OfflineVoiceCloningSimplifiedCollateFunc(
                max_audio_length=self.config.datasets.max_audio_length,
                audio_pad_token=getattr(self.config.datasets, "audio_pad_token", None),
                audio_eos_token=getattr(self.config.datasets, "audio_eos_token", None),
                text_pad_token=self.config.datasets.text_pad_token,
                audio_pad_type=self.config.datasets.audio_pad_type,
                use_eos_as_pad=self.config.datasets.use_eos_as_pad,
            )
        elif self.config.datasets.type == "hf_text_tokenizer":
            print("\n\n\tUsing HF Text Tokenizer Collator for training dataloader!\n\n")

            text_tokenizer = AutoTokenizer.from_pretrained(self.config.datasets.text_tokenizer_name)
            if text_tokenizer.pad_token is None:
                text_tokenizer.add_special_tokens({"pad_token": text_tokenizer.eos_token})

            collate_fn = HFTextTokenizerCollator(
                text_tokenizer=text_tokenizer,
                max_audio_length=self.config.datasets.max_audio_length,
                audio_pad_token=getattr(self.config.datasets, "audio_pad_token", None),
                audio_eos_token=getattr(self.config.datasets, "audio_eos_token", None),
                audio_pad_type=self.config.datasets.audio_pad_type,
                use_eos_as_pad=self.config.datasets.use_eos_as_pad,
            )
        elif self.config.datasets.type == "phoneme_tokenizer":
            print("\n\n\tUsing Phoneme Tokenizer Collator for training dataloader!\n\n")

            phoneme_tokenizer = PhonemeTokenizer()
            collate_fn = PhonemeTokenizerCollator(
                phoneme_tokenizer=phoneme_tokenizer,
                max_audio_length=self.config.datasets.max_audio_length,
                audio_pad_token=getattr(self.config.datasets, "audio_pad_token", None),
                audio_eos_token=getattr(self.config.datasets, "audio_eos_token", None),
                audio_pad_type=self.config.datasets.audio_pad_type,
                use_eos_as_pad=self.config.datasets.use_eos_as_pad,
            )
        else:
            raise ValueError(f"Invalid dataset type: {self.config.datasets.type}")

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=self.config.train.shuffle,
            num_workers=self.config.train.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        if self.config.datasets.type == "dynamic":
            collate_fn = DynamicSingleSpeakerCollateFunc()
        elif self.config.datasets.type == "offline":
            collate_fn = OfflineMultipleSpeakerMaskCollateFunc(
                max_audio_length=self.config.datasets.max_audio_length,
                mask_prob=self.config.datasets.mask_prob,
                audio_mask_token=self.config.datasets.audio_mask_token,
                audio_pad_token=getattr(self.config.datasets, "audio_pad_token", None),
                audio_eos_token=self.config.datasets.audio_eos_token,
                text_pad_token=self.config.datasets.text_pad_token,
                mask_type=self.config.datasets.mask_type,
                audio_pad_type=self.config.datasets.audio_pad_type,
                use_eos_as_pad=self.config.datasets.use_eos_as_pad,
                loss_on_eos_pad=self.config.datasets.loss_on_eos_pad,
                pad_loss_weight=getattr(self.config.datasets, "pad_loss_weight", 1.0),
            )
        elif self.config.datasets.type == "offline_dynamic_dur":
            print("Creating collate function for offline_dynamic_dur")
            collate_fn = OfflineMultipleSpeakerDreamOnCollateFunc(
                max_audio_length=self.config.datasets.max_audio_length,
                audio_mask_token=self.config.datasets.audio_mask_token,
                audio_pad_token=getattr(self.config.datasets, "audio_pad_token", None),
                audio_expand_token=getattr(self.config.datasets, "audio_expand_token", None),
                audio_eos_token=getattr(self.config.datasets, "audio_eos_token", None),
                text_pad_token=self.config.datasets.text_pad_token,
                mask_type=self.config.datasets.mask_type,
                mask_prob=self.config.datasets.mask_prob,
                audio_pad_type=self.config.datasets.audio_pad_type,
                # Dynamic Duration Params
                mix_ratio=self.config.datasets.mix_ratio,
                p_merge_static=self.config.datasets.p_merge_static,
                p_merge_dynamic_scale=self.config.datasets.p_merge_dynamic_scale,
                delete_frac_range=self.config.datasets.delete_frac_range,
                # padding params
                use_eos_as_pad=self.config.datasets.use_eos_as_pad,
                loss_on_eos_pad=self.config.datasets.loss_on_eos_pad,
                pad_loss_weight=getattr(self.config.datasets, "pad_loss_weight", 1.0),
            )
        elif self.config.datasets.type == "offline_voice_cloning_simplified":
            collate_fn = OfflineVoiceCloningSimplifiedCollateFunc(
                max_audio_length=self.config.datasets.max_audio_length,
                audio_pad_token=getattr(self.config.datasets, "audio_pad_token", None),
                audio_eos_token=getattr(self.config.datasets, "audio_eos_token", None),
                text_pad_token=self.config.datasets.text_pad_token,
                audio_pad_type=self.config.datasets.audio_pad_type,
                use_eos_as_pad=self.config.datasets.use_eos_as_pad,
            )
        elif self.config.datasets.type == "hf_text_tokenizer":
            print("\n\n\tUsing HF Text Tokenizer Collator for validation dataloader!\n\n")

            text_tokenizer = AutoTokenizer.from_pretrained(self.config.datasets.text_tokenizer_name)
            if text_tokenizer.pad_token is None:
                text_tokenizer.add_special_tokens({"pad_token": text_tokenizer.eos_token})

            collate_fn = HFTextTokenizerCollator(
                text_tokenizer=text_tokenizer,
                max_audio_length=self.config.datasets.max_audio_length,
                audio_pad_token=getattr(self.config.datasets, "audio_pad_token", None),
                audio_eos_token=getattr(self.config.datasets, "audio_eos_token", None),
                audio_pad_type=self.config.datasets.audio_pad_type,
                use_eos_as_pad=self.config.datasets.use_eos_as_pad,
            )
        elif self.config.datasets.type == "phoneme_tokenizer":
            print("\n\n\tUsing Phoneme Tokenizer Collator for validation dataloader!\n\n")

            phoneme_tokenizer = PhonemeTokenizer()
            collate_fn = PhonemeTokenizerCollator(
                phoneme_tokenizer=phoneme_tokenizer,
                max_audio_length=self.config.datasets.max_audio_length,
                audio_pad_token=getattr(self.config.datasets, "audio_pad_token", None),
                audio_eos_token=getattr(self.config.datasets, "audio_eos_token", None),
                audio_pad_type=self.config.datasets.audio_pad_type,
                use_eos_as_pad=self.config.datasets.use_eos_as_pad,
            )
        else:
            raise ValueError(f"Invalid dataset type: {self.config.datasets.type}")

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

        if self.config.optimizer.name.lower() == "adam":
            print("Using Adam optimizer")
            optimizer = Adam(
                [
                    {"params": gain_or_bias_params, "weight_decay": 0.},
                    {"params": rest_params, "weight_decay": opt_params["weight_decay"]},
                ],
                lr=opt_params["learning_rate"],
                eps=opt_params["eps"],
                betas=opt_params["betas"],
                weight_decay=opt_params["weight_decay"]
            )

        elif self.config.optimizer.name.lower() == "adamw":
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

        elif self.config.optimizer.name.lower() == "lion":
            optimizer = Lion(
                [
                    {"params": gain_or_bias_params, "weight_decay": 0.},
                    {"params": rest_params, "weight_decay": opt_params["weight_decay"]},
                ],
                lr=opt_params["learning_rate"],
                betas=opt_params["betas"],
                weight_decay=opt_params["weight_decay"],
                use_triton=opt_params.get("use_triton", False),
            )

        else:
            raise ValueError(f"Invalid optimizer: {self.config.optimizer.name}")

        if not self.config["scheduler"]:
            return optimizer

        scheduler = None
        if self.config.scheduler.name.lower() == "reducelronplateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                "min",
                patience=scheduler_params.get("patience", max_num_steps*0.25),
                factor=0.9,
                min_lr=opt_params.get("min_learning_rate", 1.0e-6)
            )

        if self.config.scheduler.name.lower() == "cosinewarmuplr":
            scheduler = CosineWarmupLR(
                optimizer,
                lr_min=opt_params.get("min_learning_rate", 1.0e-6),
                lr_max=opt_params["learning_rate"],
                warmup=scheduler_params.get("warmup_lr", max_num_steps*0.05),
                T_max=max_num_steps
            )

        if self.config.scheduler.name.lower() == "linearlr":
            scheduler = LinearLR(
                optimizer,
                start_factor=scheduler_params.get("start_factor", 1.0 / 3.0),
                end_factor=scheduler_params.get("end_factor", 1.0),
                total_iters=scheduler_params.get("total_iters", 5),
                last_epoch=scheduler_params.get("last_epoch", -1),
                verbose=scheduler_params.get("verbose", False)
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

    def get_speech_token(self, input_waveform, input_features):
        """
        Extract speech token sequence using the encoder.
        It is assumed that encoder.encode_batch_feats returns a tensor whose shape could be (B, 1, seq_len) or (B, seq_len).
        If the returned shape is (B, 1, seq_len), squeeze out the 1st dimension.
        """
        with torch.no_grad():
            speech_tokens = self.audio_codec.encode_batch_feats(
                input_waveform=input_waveform,
                input_features=input_features
            )
        if speech_tokens.dim() == 3 and speech_tokens.size(1) == 1:
            speech_tokens = speech_tokens.squeeze(1)
        return speech_tokens.long()

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
        if self.config.datasets.type == "dynamic":
            input_waveform, input_features, transcription_ids = batch
            x_1 = self.get_speech_token(input_waveform, input_features)
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
                aux_ce_loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    ce_target.view(-1).long(),
                    ignore_index=-100,
                    reduction="mean",
                )
            self.log("train/aux_ce_loss", aux_ce_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        else:
            # use audio mask to transform padding positions to -100 so that they are ignored in the CE loss
            target = x_1.masked_fill(~audio_att_mask.bool(), -100)
            loss = self.criteria(
                input=logits.view(-1, logits.size(-1)),
                target=target.view(-1).long()
            ).mean()

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if self.config.datasets.type == "dynamic":
            input_waveform, input_features, transcription_ids = batch
            x_1 = self.get_speech_token(input_waveform, input_features)
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
                aux_ce_loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    ce_target.view(-1).long(),
                    ignore_index=-100,
                    reduction="mean",
                )
            self.log("val/aux_ce_loss", aux_ce_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        else:
            # use audio mask to transform padding positions to -100 so that they are ignored in the CE loss
            target = x_1.masked_fill(~audio_att_mask.bool(), -100)
            loss = self.criteria(
                input=logits.view(-1, logits.size(-1)),
                target=target.view(-1).long()
            ).mean()

        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if batch_idx == 0:
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

        if self.config.datasets.codec_name == "xcodec2":
            audio_codec = XCodec2Model.from_pretrained(self.config.datasets.audio_codec).to(self.device)
            saving_sr = audio_codec.config.sampling_rate
        elif self.config.datasets.codec_name == "neucodec":
            audio_codec = NeuCodec.from_pretrained("neuphonic/neucodec").to(self.device)
            saving_sr = 24000
            audio_codec.eval()
        else:
            raise ValueError(f"Invalid codec name: {self.config.datasets.codec_name}")

        audio_ref, audio_ref_sr = torchaudio.load(audio_ref_path)

        if audio_ref_sr != self.config.datasets.sampling_rate:
            audio_ref = torchaudio.transforms.Resample(audio_ref_sr, self.config.datasets.sampling_rate)(audio_ref)

        print(f"Audio reference shape: {audio_ref.shape}")

        if self.config.datasets.codec_name == "xcodec2":
            codes_ref = audio_codec.encode_code(input_waveform=audio_ref).squeeze()
        elif self.config.datasets.codec_name == "neucodec":
            codes_ref = audio_codec.encode_code(audio_ref[None, ...]).squeeze()
            print("-"*100)
            print(f"Codes reference after encoding shape: {codes_ref.shape}")
        codes_ref_size = codes_ref.shape[-1]

        print(f"Codes reference shape: {codes_ref_size}")

        print(f"Padded codes reference shape: {codes_ref.shape}")
        codes_ref = codes_ref.unsqueeze(0).to(self.device)

        if self.config.datasets.type == "hf_text_tokenizer":
            text_tokenizer = AutoTokenizer.from_pretrained(self.config.datasets.text_tokenizer_name)
            if text_tokenizer.pad_token is None:
                text_tokenizer.add_special_tokens({"pad_token": text_tokenizer.eos_token})
        elif self.config.datasets.type == "voice_bpe_tokenizer":
            text_tokenizer = VoiceBpeTokenizer(vocab_file=self.config.datasets.vocab_file)
        elif self.config.datasets.type == "phoneme_tokenizer":
            text_tokenizer = PhonemeTokenizer()
        else:
            raise ValueError(f"Invalid text tokenizer type: {self.config.datasets.type}")

        vocab_size = self.config.datasets.audio_vocab_size + self.config.model.audio_add_token
        max_length = self.config.test.max_audio_length
        generated_audios = {}
        # Iterate over each test sentence from config
        for idx, sentence in tqdm(enumerate(self.config.test.sentences), total=len(self.config.test.sentences)):
            print(f"\nGenerating audio for sentence: {sentence}")
            augmented_sentence = text_ref + " " + sentence

            if isinstance(text_tokenizer, VoiceBpeTokenizer):
                text_ids = torch.tensor(text_tokenizer.encode(augmented_sentence, lang="en-us")).to(self.device).unsqueeze(0)
            elif isinstance(text_tokenizer, PhonemeTokenizer):
                print("\n\nUsing Phoneme Tokenizer for text encoding!\n\n")
                print(f"Phonemes: {augmented_sentence}")
                text_ids = text_tokenizer(augmented_sentence).input_ids.to(self.device)
                print(f"Text IDs: {text_ids.shape}", torch.min(text_ids), torch.max(text_ids))
            else:
                print("\n\nUsing HF AutoTokenizer for text encoding!\n\n")
                text_ids = text_tokenizer(
                    augmented_sentence,
                    return_tensors="pt"
                )["input_ids"].squeeze(0).to(self.device).unsqueeze(0)

                print(text_ids)

            print(f"Text IDs: {text_ids.shape}", torch.min(text_ids), torch.max(text_ids))
            # Initialize xt with mask token (batch size = 1)
            x_t = self.source_distribution.sample((1, max_length), device=self.device)
            print("11111", x_t)
            print(f"Initial x_t: {x_t.shape}, {torch.min(x_t)}, {torch.max(x_t)}")
            print(f"Initial codes_ref: {codes_ref.shape}, {torch.min(codes_ref)}, {torch.max(codes_ref)}")
            print(f"Initial text_ids: {text_ids.shape}, {torch.min(text_ids)}, {torch.max(text_ids)}")

            if self.config.datasets.cond_drop_prob==0.0:
                print("\n\tUsing simple_generate_sample_ctmc\n")
                x_t = self.simple_generate_sample_ctmc(
                    xt=x_t,
                    text_ids=text_ids,
                    codes_ref=codes_ref,
                    nsf=self.config.test.nsf,
                    codes_ref_size=codes_ref_size
                )
            else:
                if self.config.source_dist_type == "mask":
                    print("\n\tUsing PFG generator with Masked Source Distribution\n")
                    x_t = self.generate_sample_pfg_mask(
                        xt=x_t,
                        text_ids=text_ids,
                        codes_ref=codes_ref,
                        nsf=self.config.test.nsf,
                        codes_ref_size=codes_ref_size
                    )
                else:
                    print("\n\tUsing PFG generator with Uniform Source Distribution\n")
                    x_t = self.generate_sample_pfg_euler_uniform(
                        xt=x_t,
                        text_ids=text_ids,
                        codes_ref=codes_ref,
                        nsf=self.config.test.nsf,
                        codes_ref_size=codes_ref_size
                    )

            codec_vocab = int(self.config.datasets.audio_vocab_size)  # 65536
            mx = int(x_t.max().item())
            mn = int(x_t.min().item())
            print(f"[DEBUG] decode input token range: min={mn} max={mx} codec_vocab={codec_vocab}")
            if mx >= codec_vocab:
                bad = (x_t >= codec_vocab).sum().item()
                print(f"[DEBUG] WARNING: {bad} tokens >= codec_vocab (these will crash decode_code)")
                print("Removing bad tokens...")
                # remove the tokens that are >= codec_vocab, and put the elements on the left
                x_t = x_t[x_t < codec_vocab]
            generated_audio = audio_codec.decode_code(x_t)
            # Use a truncated version of the sentence for the log key (replace spaces with underscores)
            key = f"generated_audio_{idx}"
            generated_audios[key] = wandb.Audio(
                generated_audio[0, 0, :].cpu().numpy(),
                sample_rate=saving_sr,
                caption=sentence
            )
        wandb.log(generated_audios)

    @torch.no_grad()
    def simple_generate_sample_paper(self, xt, text_ids, codes_ref, nsf: int, codes_ref_size: int):
        num_steps = nsf
        dt = 1.0 / num_steps
        S = self.config.datasets.audio_vocab_size + self.config.model.audio_add_token
        eps = 1e-12

        xt = xt.clone()
        xt[..., :codes_ref_size] = codes_ref[..., :codes_ref_size]

        text_att_mask = torch.ones((xt.size(0), text_ids.size(1)), device=xt.device, dtype=torch.bool)
        audio_att_mask = torch.ones_like(xt, device=xt.device, dtype=torch.bool)

        pad_id = getattr(self.config.datasets, "audio_pad_token", None)

        for step in range(num_steps):
            t_val = step / num_steps
            t_tensor = torch.full((xt.size(0),), t_val, device=xt.device, dtype=torch.float32)

            logits = self(
                x_t=xt,
                text_ids=text_ids,
                text_att_mask=text_att_mask,
                audio_att_mask=audio_att_mask,
                time=t_tensor,
                drop_text=False,
            ).float()

            if logits.size(-1) != S:
                raise RuntimeError(f"logits classes {logits.size(-1)} != V_total {S}")

            if pad_id is not None:
                logits[..., int(pad_id)] = -torch.inf

            p1 = torch.softmax(logits, dim=-1)

            # final step: sample directly from p_{1|t} (at t ~ 1 - 1/nsf)
            if step == num_steps - 1:
                xt = torch.distributions.Categorical(probs=p1).sample()
                xt[..., :codes_ref_size] = codes_ref[..., :codes_ref_size]
                break

            sched = self.path.scheduler(t_tensor)
            kappa  = sched.alpha_t          # tensor [B]
            dkappa = sched.d_alpha_t        # tensor [B]

            one_hot_x_t = torch.nn.functional.one_hot(xt, num_classes=S).float()

            rate = (dkappa / (1.0 - kappa).clamp_min(eps)).view(-1, 1, 1)
            u = rate * (p1 - one_hot_x_t)

            new_probs = (one_hot_x_t + dt * u).clamp_min(0)
            new_probs = new_probs / new_probs.sum(dim=-1, keepdim=True).clamp_min(eps)

            xt = torch.distributions.Categorical(probs=new_probs).sample()
            xt[..., :codes_ref_size] = codes_ref[..., :codes_ref_size]

        # --- match your other samplers: remove the prompt prefix ---
        xt = xt[..., codes_ref_size:]

        # cleanup (consider truncating at first EOS instead of removing all EOS)
        xt = xt.squeeze(0)
        xt = xt[xt != self.config.datasets.audio_mask_token]
        if hasattr(self.config.datasets, "audio_eos_token"):
            xt = xt[xt != self.config.datasets.audio_eos_token]
        if hasattr(self.config.datasets, "audio_pad_token"):
            xt = xt[xt != self.config.datasets.audio_pad_token]

        return xt.unsqueeze(0).unsqueeze(0)

    @torch.no_grad()
    def simple_generate_sample_ctmc(
        self,
        xt,
        text_ids,
        codes_ref,
        nsf: int,
        codes_ref_size: int,
    ):
        """
        CTMC jump-or-stay sampler equivalent (up to helper details) to
        flow_matching.solver.MixtureDiscreteEulerSolver with div_free = 0.

        Key differences vs your Euler-on-simplex sampler:
        - sample x1 ~ p_{1|t}
        - build jump rates u = (dκ/(1-κ)) * 1[x == x1], then set u(x_t)=0
        - jump with prob 1 - exp(-dt * intensity); if jump sample from u (normalized)
        """
        num_steps = int(nsf)
        dt = 1.0 / max(num_steps, 1)
        S = int(self.config.datasets.audio_vocab_size + self.config.model.audio_add_token)
        eps = 1e-12

        xt = xt.clone()
        xt[..., :codes_ref_size] = codes_ref[..., :codes_ref_size]

        # attention masks (adjust if you actually have padding)
        text_att_mask = torch.ones(
            (xt.size(0), text_ids.size(1)),
            device=xt.device,
            dtype=torch.bool,
        )
        audio_att_mask = torch.ones_like(xt, device=xt.device, dtype=torch.bool)

        pad_id = getattr(self.config.datasets, "audio_pad_token", None)

        for step in range(num_steps):
            t_val = step / num_steps
            t_tensor = torch.full(
                (xt.size(0),),
                t_val,
                device=xt.device,
                dtype=torch.float32,
            )

            logits = self(
                x_t=xt,
                text_ids=text_ids,
                text_att_mask=text_att_mask,
                audio_att_mask=audio_att_mask,
                time=t_tensor,
                drop_text=False,
            ).float()

            if logits.size(-1) != S:
                raise RuntimeError(f"logits classes {logits.size(-1)} != V_total {S}")

            # Prevent sampling PAD as x1 / final sample (optional, but usually desired)
            if pad_id is not None:
                logits[..., int(pad_id)] = -torch.inf

            p1 = torch.softmax(logits, dim=-1)

            # Final step: x_t <- x1 ~ p_{1|t}
            if step == num_steps - 1:
                xt = torch.distributions.Categorical(probs=p1).sample()
                xt[..., :codes_ref_size] = codes_ref[..., :codes_ref_size]
                break

            # --- CTMC step (MixtureDiscreteEulerSolver with div_free=0) ---

            # 1) sample x1 ~ p_{1|t}(. | x_t)
            x1 = torch.distributions.Categorical(probs=p1).sample()  # [B, T]

            # 2) get scheduler values
            sched = self.path.scheduler(t_tensor)
            kappa = sched.alpha_t          # [B]
            dkappa = sched.d_alpha_t       # [B]

            rate = (dkappa / (1.0 - kappa).clamp_min(eps)).view(-1, 1, 1)  # [B,1,1]

            # 3) build off-diagonal jump rates u
            # u(x, x_t | x1) = rate * 1[x == x1], then set u(x_t)=0
            delta1 = torch.nn.functional.one_hot(x1, num_classes=S).to(rate.dtype)  # [B,T,S]
            u = rate * delta1  # [B,T,S]

            # set u(x_t)=0 (no self-transition)
            delta_t = torch.nn.functional.one_hot(xt, num_classes=S).to(torch.bool)  # [B,T,S]
            u = torch.where(delta_t, torch.zeros_like(u), u)

            # OPTIONAL but recommended: freeze prompt prefix so it never jumps
            if codes_ref_size > 0:
                u[:, :codes_ref_size, :] = 0.0

            # 4) jump-or-stay
            intensity = u.sum(dim=-1)  # [B,T]
            # P(jump) = 1 - exp(-dt * intensity)
            p_jump = 1.0 - torch.exp(-dt * intensity)
            jump = torch.rand_like(p_jump) < p_jump  # [B,T] boolean

            if jump.any():
                # sample new token from u (normalized) at jumped positions
                u_jump = u[jump]  # [N,S]
                probs_jump = u_jump / u_jump.sum(dim=-1, keepdim=True).clamp_min(eps)
                new_tokens = torch.distributions.Categorical(probs=probs_jump).sample()  # [N]
                xt = xt.clone()
                xt[jump] = new_tokens

            # re-impose fixed prefix
            xt[..., :codes_ref_size] = codes_ref[..., :codes_ref_size]

        # --- match your other samplers: remove the prompt prefix ---
        xt = xt[..., codes_ref_size:]

        # cleanup (same as you had)
        xt = xt.squeeze(0)
        xt = xt[xt != self.config.datasets.audio_mask_token]
        if hasattr(self.config.datasets, "audio_eos_token"):
            xt = xt[xt != self.config.datasets.audio_eos_token]
        if hasattr(self.config.datasets, "audio_pad_token"):
            xt = xt[xt != self.config.datasets.audio_pad_token]

        return xt.unsqueeze(0).unsqueeze(0)


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

            if self.config.datasets.type == "offline_dynamic_dur":
                xt = self.apply_vlg_ops(
                    x=xt,
                    mask_token=self.config.datasets.audio_mask_token,
                    expand_token=self.config.datasets.audio_expand_token,
                    # delete_token=self.config.datasets.audio_delete_token,
                    delete_token=self.config.datasets.audio_eos_token,
                    max_len=self.config.test.max_audio_length,
                    edit_start=codes_ref_size,
                    edit_end=None
                )
                if codes_ref.size(1) < xt.size(1):
                    codes_ref = F.pad(codes_ref, (0, xt.size(1) - codes_ref.size(1)), value=self.config.datasets.audio_mask_token)
                elif codes_ref.size(1) > xt.size(1):
                    codes_ref = codes_ref[:, :xt.size(1)]

            # Force conditioning
            xt[..., : codes_ref_size] = codes_ref[..., : codes_ref_size]

        print("Final xt shape:", xt.shape)
        xt = xt[..., codes_ref_size:]
        print("Shape after removing codes_ref:", xt.shape)
        # remove making tokens from the generated sequence
        xt = xt.squeeze(0)
        print("Shape after squeeze:", xt.shape)
        # Trim to the first EOS (if present)
        # eos_pos = (xt == self.config.datasets.audio_eos_token).nonzero(as_tuple=False).squeeze(-1)
        # if eos_pos.numel() > 0:
        #     first_eos = int(eos_pos[0].item())
        #     xt = xt[..., :first_eos]
        xt = xt[xt != self.config.datasets.audio_eos_token]
        print(f"Shape after eos removal:", xt.shape)
        xt = xt[xt != self.config.datasets.audio_mask_token]
        print("Shape after mask removal:", xt.shape)

        # remove padding tokens from the generated sequence
        if hasattr(self.config.datasets, "audio_pad_token"):
            # remove padding tokens from the generated sequence
            xt = xt[xt != self.config.datasets.audio_pad_token]
            print("Shape after pad removal:", xt.shape)

        if hasattr(self.config.datasets, "audio_expand_token"):
            # remove padding tokens from the generated sequence
            xt = xt[xt != self.config.datasets.audio_expand_token]
            print("Shape after expand removal:", xt.shape)

        if hasattr(self.config.datasets, "audio_delete_token"):
            # remove padding tokens from the generated sequence
            xt = xt[xt != self.config.datasets.audio_delete_token]
            print("Shape after delete removal:", xt.shape)

        print("Shape after pad removal:", xt.shape)
        if self.config.datasets.type == "offline_dynamic_dur":
            xt = xt[xt != self.config.datasets.audio_expand_token]
            print("Shape after expand removal:", xt.shape)
            # xt = xt[xt != self.config.datasets.audio_delete_token]
            # print("Shape after delete removal:", xt.shape)
        xt = xt.unsqueeze(0).unsqueeze(0)
        print("Final Shape", xt.shape)

        return xt

    def generate_sample_pfg_uniform(self, xt, text_ids, codes_ref, nsf: int, codes_ref_size: int):
        """
        Discrete CTMC sampler with predictor-free guidance (uniform base).
        Differences vs mask-absorbing:
        - No special-case gating on mask tokens.
        - Off-diagonal rates exist for every position at all times.
        - Diagonal is set by negative row-sum of off-diagonals.
        """
        num_steps = nsf
        dt = 1.0 / max(1, num_steps)
        x1_temp = 1.0 # sampling temperature for logits -> probs
        gamma = 2.5 # PFG guidance scale
        S = self.config.datasets.audio_vocab_size + self.config.model.audio_add_token
        eps = 1e-9
        noise = 0.0

        # Optional config shorthands for cleanup
        mask_token_id   = getattr(self.config.datasets, "audio_mask_token", None)
        eos_token_id    = getattr(self.config.datasets, "audio_eos_token", None)
        pad_token_id    = getattr(self.config.datasets, "audio_pad_token", None)
        expand_token_id = getattr(self.config.datasets, "audio_expand_token", None)
        delete_token_id = getattr(self.config.datasets, "audio_delete_token", None)

        # Force-reference (prefix) constraint
        xt[..., : codes_ref_size] = codes_ref[..., : codes_ref_size]

        # Create text att_mask (single sample)
        text_att_mask = text_ids.new_ones((1, text_ids.size(1)), dtype=torch.bool)

        # Time loop
        for step in tqdm(range(num_steps), total=num_steps):
            t_val    = step * dt
            t_tensor = xt.new_full((1,), t_val, dtype=torch.float32, device=self.device)

            # ----- Unconditional pass -----
            logits_u = self(
                x_t=xt,
                text_ids=text_ids,
                text_att_mask=text_att_mask,
                time=t_tensor,
                drop_text=True,
            )
            probs_u = torch.softmax(logits_u / x1_temp, dim=-1)  # [B, T, S]

            # ----- Conditional pass -----
            logits_c = self(
                x_t=xt,
                text_ids=text_ids,
                text_att_mask=text_att_mask,
                time=t_tensor,
                drop_text=False,
            )
            probs_c = torch.softmax(logits_c / x1_temp, dim=-1)  # [B, T, S]

            # ban_ids = []
            # for attr in ["audio_pad_token"]:
            #     tid = getattr(self.config.datasets, attr, None)
            #     if tid is not None:
            #         ban_ids.append(int(tid))
            # if ban_ids:
            #     ban_mask = torch.zeros(S, device=xt.device, dtype=probs_u.dtype)
            #     ban_mask[torch.tensor(ban_ids, device=xt.device)] = 1.0
            #     # zero banned columns before removing the diagonal & renorm
            #     probs_u = probs_u * (1.0 - ban_mask.view(1,1,S))
            #     probs_c = probs_c * (1.0 - ban_mask.view(1,1,S))

            # ----- Build uniform-base CTMC rates -----
            # Hazard / noise-rate schedule
            base_r = (1.0 + noise * t_val) / max(eps, (1.0 - t_val))  # scalar λ(t) > 0

            # Remove self probability (no instantaneous self-jumps)
            one_hot_cur = F.one_hot(xt, num_classes=S).float() # [B, T, S]
            # Off-diagonal jump distributions (normalized):
            off_u = probs_u * (1.0 - one_hot_cur)
            off_u = off_u / off_u.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            off_c = probs_c * (1.0 - one_hot_cur)
            off_c = off_c / off_c.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            # Convert to instantaneous rates
            R_u = base_r * off_u # [B, T, S], off-diag ≥ 0
            R_c = base_r * off_c

            # ----- Predictor-free guidance (geometric mean of rates) -----
            log_Ru = torch.log(R_u + eps)
            log_Rc = torch.log(R_c + eps)
            R_mix  = torch.exp(gamma * log_Rc + (1.0 - gamma) * log_Ru)  # off-diag rates

            # Enforce CTMC row-sum zero by setting diagonal = -sum(off-diag)
            # First, compute off-diagonal sum (zero out diagonal temporarily)
            R_off = R_mix.clone()
            R_off.scatter_(-1, xt[..., None], 0.0)
            row_sum = R_off.sum(dim=-1, keepdim=True)
            # Fill diagonal with negative row-sum
            R = R_off.clone()
            R.scatter_(-1, xt[..., None], -row_sum)

            # ----- Euler step to a transition matrix P ≈ I + R*dt -----
            # Build P using only off-diagonals from R_off and computed diagonal
            P_off = (R_off * dt).clamp_min(0.0)
            diag = (1.0 - P_off.sum(dim=-1, keepdim=True)).clamp_min(0.0)
            P = P_off.clone()
            P.scatter_(-1, xt[..., None], diag)

            # Sample next state for each position independently
            xt = torch.multinomial(P.view(-1, S), 1).view_as(xt)

            # Force conditioning prefix after edits
            xt[..., : codes_ref_size] = codes_ref[..., : codes_ref_size]

        # ----- Post-processing / cleanup -----
        print("Final xt shape:", xt.shape)
        xt = xt[..., codes_ref_size:]
        print("Shape after removing codes_ref:", xt.shape)

        xt = xt.squeeze(0)
        print("Shape after squeeze:", xt.shape)

        # Remove EOS/MASK/PAD/EXPAND/DELETE tokens if present
        if eos_token_id is not None:
            xt = xt[xt != eos_token_id]
            print("Shape after eos removal:", xt.shape)
        if mask_token_id is not None:
            xt = xt[xt != mask_token_id]
            print("Shape after mask removal:", xt.shape)
        if pad_token_id is not None:
            xt = xt[xt != pad_token_id]
            print("Shape after pad removal:", xt.shape)
        if expand_token_id is not None:
            xt = xt[xt != expand_token_id]
            print("Shape after expand removal:", xt.shape)
        if delete_token_id is not None:
            xt = xt[xt != delete_token_id]
            print("Shape after delete removal:", xt.shape)

        xt = xt.unsqueeze(0).unsqueeze(0)
        print("Final Shape", xt.shape)
        return xt

    @torch.no_grad()
    def generate_sample_pfg_uniform_v2(self, xt, text_ids, codes_ref, nsf: int, codes_ref_size: int):
        """
        Uniform-source CTMC sampler with predictor-free guidance (PFG),
        aligned with flow_matching scheduler:
        lambda(t) = d_alpha_t / (1 - alpha_t)

        Uses jump-probability tau-leap:
        p_jump = 1 - exp(-dt * hazard)
        if jump -> sample dest from R_off / hazard

        IMPORTANT:
        - Only makes sense if you trained with source_dist_type="uniform"
        - Only use PFG if cond_drop_prob > 0 during training
        """
        num_steps = int(nsf)
        dt = 1.0 / max(1, num_steps)

        # knobs (consider exposing as args)
        x1_temp = 1.0
        guidance_scale = 2.5   # CFG-style scale (can be > 1)
        eps = 1e-9

        S = int(self.config.datasets.audio_vocab_size + self.config.model.audio_add_token)

        mask_token_id = int(getattr(self.config.datasets, "audio_mask_token", -1))
        eos_token_id  = int(getattr(self.config.datasets, "audio_eos_token", -1))

        pad_token_id = getattr(self.config.datasets, "audio_pad_token", None)
        pad_token_id = int(pad_token_id) if pad_token_id is not None else None

        # pin prefix
        xt = xt.clone()
        xt[..., :codes_ref_size] = codes_ref[..., :codes_ref_size]

        text_att_mask = text_ids.new_ones((xt.size(0), text_ids.size(1)), dtype=torch.bool)

        for step in range(num_steps):
            t_val = step * dt
            t = xt.new_full((xt.size(0),), t_val, dtype=torch.float32)

            # ---- scheduler-consistent hazard scale ----
            sched = self.path.scheduler(t)  # MixtureDiscreteProbPath scheduler
            alpha_t = sched.alpha_t
            dalpha_t = sched.d_alpha_t.clamp_min(1e-6)
            lam = (dalpha_t / (1.0 - alpha_t).clamp_min(1e-6)).view(-1, 1, 1)  # [B,1,1]

            # ---- logits ----
            logits_u = self(
                x_t=xt,
                text_ids=text_ids,
                text_att_mask=text_att_mask,
                time=t,
                drop_text=True,
            ).float()
            logits_c = self(
                x_t=xt,
                text_ids=text_ids,
                text_att_mask=text_att_mask,
                time=t,
                drop_text=False,
            ).float()

            # Optional: ban PAD if it exists and is not EOS-as-PAD
            if pad_token_id is not None and not bool(getattr(self.config.datasets, "use_eos_as_pad", False)):
                logits_u[..., pad_token_id] = -torch.inf
                logits_c[..., pad_token_id] = -torch.inf

            # softmax temperature (you can schedule this if you want)
            probs_u = torch.softmax(logits_u / max(1e-3, x1_temp), dim=-1)
            probs_c = torch.softmax(logits_c / max(1e-3, x1_temp), dim=-1)

            # never jump *into* MASK token in uniform-base validation
            if 0 <= mask_token_id < S:
                probs_u[..., mask_token_id] = 0.0
                probs_c[..., mask_token_id] = 0.0
                probs_u = probs_u / probs_u.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                probs_c = probs_c / probs_c.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            # ---- off-diagonal jump distributions ----
            # remove self
            off_u = probs_u.clone()
            off_c = probs_c.clone()
            off_u.scatter_(-1, xt[..., None], 0.0)
            off_c.scatter_(-1, xt[..., None], 0.0)

            off_u = off_u / off_u.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            off_c = off_c / off_c.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            # ---- rates ----
            R_u = lam * off_u
            R_c = lam * off_c

            # PFG / CFG in log-rate space:
            # log R = log Ru + s * (log Rc - log Ru)
            s = float(guidance_scale)
            log_Ru = torch.log(R_u + eps)
            log_Rc = torch.log(R_c + eps)
            log_R  = log_Ru + s * (log_Rc - log_Ru)
            R_off  = torch.exp(log_R)

            # ensure diagonal = 0
            R_off.scatter_(-1, xt[..., None], 0.0)
            R_off = torch.nan_to_num(R_off, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

            # ---- tau-leap jump step ----
            hazard = R_off.sum(dim=-1)  # [B,T]
            p_jump = 1.0 - torch.exp(-dt * hazard)  # [B,T]
            do_jump = (torch.rand_like(p_jump) < p_jump)

            # never change prefix
            do_jump[..., :codes_ref_size] = False

            if do_jump.any():
                hazard_safe = hazard.clamp_min(1e-9)
                q = R_off / hazard_safe.unsqueeze(-1)  # [B,T,S]

                q2 = q.view(-1, S)
                jump_idx = do_jump.view(-1).nonzero(as_tuple=False).squeeze(-1)

                q_jump = q2.index_select(0, jump_idx)
                q_jump = torch.nan_to_num(q_jump, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
                q_jump = q_jump / q_jump.sum(dim=-1, keepdim=True).clamp_min(1e-12)

                sampled = torch.multinomial(q_jump, 1).squeeze(-1)

                xt_flat = xt.view(-1)
                xt_flat[jump_idx] = sampled.to(xt_flat.dtype)
                xt = xt_flat.view_as(xt)

            # re-pin prefix
            xt[..., :codes_ref_size] = codes_ref[..., :codes_ref_size]

        # ---- cleanup: truncate at first EOS (better than removing all EOS) ----
        xt = xt[..., codes_ref_size:]
        xt = xt.squeeze(0)

        if eos_token_id is not None and eos_token_id >= 0:
            pos = (xt == eos_token_id).nonzero(as_tuple=False)
            if pos.numel() > 0:
                xt = xt[: int(pos[0].item())]

        if mask_token_id is not None and mask_token_id >= 0:
            xt = xt[xt != mask_token_id]
        if pad_token_id is not None:
            xt = xt[xt != pad_token_id]

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
        mask_id = int(getattr(self.config.datasets, "audio_mask_token", -1))
        eos_id  = int(getattr(self.config.datasets, "audio_eos_token", -1))
        pad_id  = getattr(self.config.datasets, "audio_pad_token", None)
        pad_id  = int(pad_id) if pad_id is not None else None
        use_eos_as_pad = bool(getattr(self.config.datasets, "use_eos_as_pad", False))
        expand_id = getattr(self.config.datasets, "audio_expand_token", None)
        delete_id = getattr(self.config.datasets, "audio_delete_token", None)
        expand_id = int(expand_id) if expand_id is not None else None
        delete_id = int(delete_id) if delete_id is not None else None

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