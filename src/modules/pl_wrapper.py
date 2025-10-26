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
from lightning.pytorch.utilities import grad_norm
from xcodec2.modeling_xcodec2 import XCodec2Model
from neucodec import NeuCodec
from torch.distributions.categorical import Categorical

from modules.model import Transformer
from modules.model_cross_att import TransformerCrossAttn
from utils.tokenizer import VoiceBpeTokenizer
from dataset.build_dataset import build_dataset
from utils.lr_schedulers import CosineWarmupLR, LinearLR
from modules.flow import (
    MixtureDiscreteProbPath,
    PolynomialConvexScheduler,
    KineticOptimalMixtureScheduler,
    MaskedSourceDistribution,
    UniformSourceDistribution
)
from dataset.dataloader import (
    DynamicSingleSpeakerCollateFunc,
    OfflineMultipleSpeakerMaskCollateFunc,
    OfflineMultipleSpeakerDreamOnCollateFunc
)


class DFMTTSWrapper(L.LightningModule):
    def __init__(
        self,
        config: DictConfig,
    ):
        super().__init__()
        self.config = config

        # --- Loss config (default = plain CE) ---
        self.loss_name = getattr(getattr(config, "loss", None), "name", "ce").lower()
        # self.lambda_clip = float(getattr(getattr(config, "loss", None), "lambda_clip", 50.0))
        # self.lambda_normalize = bool(getattr(getattr(config, "loss", None), "normalize", True))

        if self.config.source_dist_type == "uniform":
            self.source_distribution = UniformSourceDistribution(
                vocab_size=self.config.datasets.audio_vocab_size
            )
        elif self.config.source_dist_type == "mask":
            self.source_distribution = MaskedSourceDistribution(
                mask_token=self.config.datasets.audio_mask_token
            )
        else:
            raise ValueError(f"Invalid source distribution: {self.config.source_dist_type}")

        if getattr(self.config, "kop_scheduler", False):
            self.path = MixtureDiscreteProbPath(
                scheduler=KineticOptimalMixtureScheduler()
            )
        else:
            self.path = MixtureDiscreteProbPath(
                scheduler=PolynomialConvexScheduler(n=self.config.datasets.n)
            )

        self.criteria = torch.nn.CrossEntropyLoss(reduction="none")

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

    # --------- KOP ELBO helpers ----------
    def _lambda_weights(self, t: torch.Tensor, ref_tokens: torch.Tensor) -> torch.Tensor:
        """
        Compute per-position λ(t) = dκ/(1-κ) from the *current* scheduler, expanded to [B, L].
        """
        sched = self.path.scheduler(t)  # alpha_t, d_alpha_t are [B]
        kappa = sched.alpha_t
        dkappa = sched.d_alpha_t
        lam = dkappa / (1.0 - kappa + 1e-12)          # [B]
        # lam = lam.clamp_max(self.lambda_clip)          # guard near t→1
        # expand to [B, L] using ref_tokens as shape guide
        return lam.view(-1, 1).expand_as(ref_tokens).to(ref_tokens.dtype)

    def _reduce_loss_ce(
        self,
        ce: torch.Tensor,
        mask_flat: torch.Tensor,
        extra_w_flat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Reduce CE over masked positions, optionally with extra weights.
        ce: [B*L], mask_flat: [B*L] bool, extra_w_flat: [B*L] or None
        """
        if extra_w_flat is not None:
            ce_m = ce[mask_flat]
            w_m = extra_w_flat[mask_flat].float().to(ce.device)
            return (ce_m * w_m).sum() / (w_m.sum() + 1e-8)
        else:
            return ce[mask_flat].mean()

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

    def forward(
        self,
        x_t: Tensor,
        text_ids: Tensor,
        text_att_mask: Tensor,
        time: Tensor,
        drop_text: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        return self.model(
            x_t=x_t,
            text=text_ids,
            text_att_mask=text_att_mask,
            time=time,
            drop_text=drop_text,
        )

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

    def training_step(self, batch, batch_idx):
        if self.config.datasets.type == "dynamic":
            input_waveform, input_features, transcription_ids = batch
            x_1 = self.get_speech_token(input_waveform, input_features)
        elif self.config.datasets.type == "offline" or \
            self.config.datasets.type == "offline_dynamic_dur":
            if len(batch) == 6:
                x_1, transcription_ids, transcription_att_mask, x_0, audio_att_mask, loss_weight_extra = batch
            else:
                x_1, transcription_ids, transcription_att_mask, x_0, audio_att_mask = batch
                loss_weight_extra = None

        with torch.no_grad():
            t = torch.rand(x_1.shape[0], device=x_1.device)
            path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)

            x_t = path_sample.x_t
            times_t = path_sample.t

            m_corrupt = (x_t != x_1) & audio_att_mask.bool()

        if random.random() < self.config.datasets.cond_drop_prob:
            drop_text = True
        else:
            drop_text = False

        # Assert to verify correctness when the conditional drop probability is set to 0
        if self.config.datasets.assert_drop_prob and self.config.datasets.cond_drop_prob == 0:
            assert drop_text == False , "Drop text should be False for training when cond_drop_prob is 0"

        # print(f"TRAINER - TRAIN: {times_t}")
        logits = self(
            x_t=x_t,
            text_ids=transcription_ids,
            text_att_mask=transcription_att_mask,
            time=times_t,
            drop_text=drop_text,
        )

        ce = self.criteria(logits.flatten(0, 1), x_1.flatten(0, 1).long())
        m  = m_corrupt.flatten(0, 1).bool()

        # Apply mask to the loss
        # if loss_weight_extra is not None:
        #     w = loss_weight_extra.flatten(0, 1).float().to(ce.device)  # [B*L]
        #     ce_m = ce[m]
        #     w_m = w[m]
        #     loss = (ce_m * w_m).sum() / (w_m.sum() + 1e-8)
        # else:
        #     loss = ce[m].mean()

        # ---- Loss: CE or KOP-ELBO (mixture) ----
        if self.loss_name == "kop_elbo":
            # print("Using KOP-ELBO loss")
            # per-position λ(t)
            lam = self._lambda_weights(times_t, ref_tokens=x_1)        # [B, L]
            lam_flat = lam.flatten(0, 1)
            # optionally combine with provided pad/EOS weights
            if loss_weight_extra is not None:
                w_extra = loss_weight_extra.flatten(0, 1).float().to(ce.device)
                w = lam_flat * w_extra
            else:
                w = lam_flat
            # optional normalization to keep loss scale stable across schedules
            # if self.lambda_normalize:
            #     loss = self._reduce_loss_ce(ce, m, w)
            # else:
            # unnormalized weighted mean over masked
            loss = (ce[m] * w[m]).mean()
        else:
            flattened_audio_att_mask = audio_att_mask.flatten(0, 1).bool()
            if loss_weight_extra is not None:
                w = loss_weight_extra.flatten(0, 1).float().to(ce.device)
                loss = self._reduce_loss_ce(ce, flattened_audio_att_mask, w)
            else:
                loss = ce[flattened_audio_att_mask].mean()

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if self.config.datasets.type == "dynamic":
            input_waveform, input_features, transcription_ids = batch
            x_1 = self.get_speech_token(input_waveform, input_features)
        elif self.config.datasets.type == "offline" or \
            self.config.datasets.type == "offline_dynamic_dur":
            if len(batch) == 6:
                x_1, transcription_ids, transcription_att_mask, x_0, audio_att_mask, loss_weight_extra = batch
            else:
                x_1, transcription_ids, transcription_att_mask, x_0, audio_att_mask = batch
                loss_weight_extra = None

        with torch.no_grad():
            t = torch.rand(x_1.shape[0], device=x_1.device)
            path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)

            x_t = path_sample.x_t
            times_t = path_sample.t

            m_corrupt = (x_t != x_1) & audio_att_mask.bool()

        # print(f"TRAINER - VAL: {times_t}")
        logits = self(
            x_t=x_t,
            text_ids=transcription_ids,
            text_att_mask=transcription_att_mask,
            time=times_t,
            drop_text=False,
        )

        ce = self.criteria(logits.flatten(0, 1), x_1.flatten(0, 1).long())
        # m  = mask.flatten(0, 1).bool()
        # Apply mask to the loss
        # if loss_weight_extra is not None:
        #     w = loss_weight_extra.flatten(0, 1).float().to(ce.device)  # [B*L]
        #     ce_m = ce[m]
        #     w_m = w[m]
        #     loss = (ce_m * w_m).sum() / (w_m.sum() + 1e-8)
        # else:
        #     loss = ce[m].mean()
        # ---- Loss: CE or KOP-ELBO (mixture) ----
        if self.loss_name == "kop_elbo":
            m  = m_corrupt.flatten(0, 1).bool()
            # per-position λ(t)
            lam = self._lambda_weights(times_t, ref_tokens=x_1)        # [B, L]
            lam_flat = lam.flatten(0, 1)
            # optionally combine with provided pad/EOS weights
            if loss_weight_extra is not None:
                w_extra = loss_weight_extra.flatten(0, 1).float().to(ce.device)
                w = lam_flat * w_extra
            else:
                w = lam_flat
            # optional normalization to keep loss scale stable across schedules
            # if self.lambda_normalize:
            #     loss = self._reduce_loss_ce(ce, m, w)
            # else:
            # unnormalized weighted mean over masked
            loss = (ce[m] * w[m]).mean()
        else:
            flattened_audio_att_mask = audio_att_mask.flatten(0, 1).bool()
            if loss_weight_extra is not None:
                w = loss_weight_extra.flatten(0, 1).float().to(ce.device)
                loss = self._reduce_loss_ce(ce, flattened_audio_att_mask, w)
            else:
                loss = ce[flattened_audio_att_mask].mean()

        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if batch_idx == 0:
            try:
                self.sample_validation()
            except Exception as e:
                print(f"Error during validation sample generation: {e}")
                wandb.log({"validation_sample": None})
                pass
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

        # pad codes_ref to have the same length as the model's max_length
        if codes_ref.size(0) < self.config.test.max_audio_length:
            codes_ref = F.pad(
                codes_ref,
                (0, self.config.test.max_audio_length - codes_ref.size(0)),
                value=self.config.datasets.audio_mask_token
            )

        print(f"Padded codes reference shape: {codes_ref.shape}")
        codes_ref = codes_ref.unsqueeze(0).to(self.device)

        text_tokenizer = VoiceBpeTokenizer(vocab_file=self.config.datasets.vocab_file)

        vocab_size = self.config.datasets.audio_vocab_size + self.config.model.add_token
        max_length = self.config.test.max_audio_length
        generated_audios = {}
        # Iterate over each test sentence from config
        for idx, sentence in tqdm(enumerate(self.config.test.sentences), total=len(self.config.test.sentences)):
            print(f"\nGenerating audio for sentence: {sentence}")
            augmented_sentence = text_ref + " " + sentence
            text_ids = torch.tensor(text_tokenizer.encode(augmented_sentence, lang="en-us")).to(self.device).unsqueeze(0)
            print(f"Text IDs: {text_ids.shape}", torch.min(text_ids), torch.max(text_ids))
            # Initialize xt with mask token (batch size = 1)
            x_t = self.source_distribution.sample((1, max_length), device=self.device)
            print("11111", x_t)
            print(f"Initial x_t: {x_t.shape}, {torch.min(x_t)}, {torch.max(x_t)}")
            print(f"Initial codes_ref: {codes_ref.shape}, {torch.min(codes_ref)}, {torch.max(codes_ref)}")
            print(f"Initial text_ids: {text_ids.shape}, {torch.min(text_ids)}, {torch.max(text_ids)}")

            if self.config.datasets.ref_drop_prob==0 and \
                self.config.datasets.cond_drop_prob==0:
                print("\n\tUsing simple_generate_sample\n")
                x_t = self.simple_generate_sample(
                    xt=x_t,
                    text_ids=text_ids,
                    codes_ref=codes_ref,
                    nsf=self.config.test.nsf,
                    codes_ref_size=codes_ref_size
                )
            else:
                print("\n\tUsing PFG generator\n")
                x_t = self.generate_sample(
                    xt=x_t,
                    text_ids=text_ids,
                    codes_ref=codes_ref,
                    nsf=self.config.test.nsf,
                    codes_ref_size=codes_ref_size
                )
            generated_audio = audio_codec.decode_code(x_t)
            # Use a truncated version of the sentence for the log key (replace spaces with underscores)
            key = f"generated_audio_{idx}"
            generated_audios[key] = wandb.Audio(
                generated_audio[0, 0, :].cpu().numpy(),
                sample_rate=saving_sr,
                caption=sentence
            )
        wandb.log(generated_audios)

    def simple_generate_sample(self, xt, text_ids, codes_ref, nsf: int, codes_ref_size: int):
        num_steps = nsf
        dt = 1.0 / num_steps
        x1_temp = 1.0
        gamma = 2.5
        mask_token_id = self.config.datasets.audio_mask_token
        S = self.config.datasets.audio_vocab_size + self.config.model.add_token
        eps = 1e-12
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
            # print(f"\n\n\t SIZE S: {S} | {xt.shape} | {torch.min(xt)}, {torch.max(xt)}")
            assert torch.min(xt) >= 0 and torch.max(xt) < S, f"xt values should be in [0, {S}), but got min {torch.min(xt)} and max {torch.max(xt)}"
            # Get Conditional Logits
            logits = self(
                x_t=xt,
                text_ids=text_ids,
                text_att_mask=text_att_mask,
                time=t_tensor,
                drop_text=False
            )
            if logits.size(-1) != S:
                raise RuntimeError(f"logits classes {logits.size(-1)} != V_total {S}")
            # Safety: xt must be < S for one_hot
            if xt.max() >= S or xt.min() < 0:
                badmax = int(xt.max().item()); badmin = int(xt.min().item())
                raise RuntimeError(f"xt out of [0,{S-1}]: min={badmin} max={badmax}")
            p1 = torch.softmax(logits, dim=-1)
            one_hot_x_t = torch.nn.functional.one_hot(xt, num_classes=S).float()
            # Compute the velocity update using the denoiser formulation
            # Here, u = (p1 - one_hot_x_t) / (1 - t), note the small epsilon for numerical stability.
            # u = (p1 - one_hot_x_t) / (1.0 - t_val + eps)
            sched = self.path.scheduler(t_tensor)
            kappa = sched.alpha_t.item()
            dkappa = sched.d_alpha_t.item()
            u = (dkappa / (1.0 - kappa + eps)) * (p1 - one_hot_x_t)
            # Euler update: compute new probabilities and sample the updated state
            new_probs = one_hot_x_t + dt * u
            new_probs = new_probs / new_probs.sum(dim=-1, keepdim=True)
            xt = torch.distributions.Categorical(probs=new_probs).sample()
        # remove making tokens from the generated sequence
        xt = xt.squeeze(0)
        xt = xt[xt != self.config.datasets.audio_mask_token]
        if hasattr(self.config.datasets, "audio_eos_token"):
            xt = xt[xt != self.config.datasets.audio_eos_token]
        # check if audio_pad_token exist in self.config.datasets
        if hasattr(self.config.datasets, "audio_pad_token"):
            # remove padding tokens from the generated sequence
            xt = xt[xt != self.config.datasets.audio_pad_token]
        xt = xt.unsqueeze(0).unsqueeze(0)

        return xt

    def apply_vlg_ops(
        self,
        x: torch.Tensor,                  # [B, L]
        mask_token: int,
        expand_token: int,
        delete_token: int,
        max_len: int,
        edit_start: int = 0,              # first editable index (e.g., codes_ref_size)
        edit_end: Optional[int] = None,   # last editable index (exclusive); None => full length
    ):
        """
        Variable-length growth (expand) & shrink (delete), applied ONLY in [edit_start, edit_end).
        - <EXPAND>  -> replace with [MASK, MASK]
        - <DELETE>  -> remove the nearest real LEFT neighbor *and* the <DELETE> itself (no mask appended)
        Then re-pad with MASK to max_len (so new slots are fillable on later steps).
        """
        B, L = x.shape
        SENTINELS = {mask_token, expand_token, delete_token}

        out = []
        for b in range(B):
            seq = x[b].tolist()
            if edit_end is None or edit_end > len(seq):
                e_end = len(seq)
            else:
                e_end = edit_end

            new_seq = []

            # Copy prefix (non-editable head)
            if edit_start > 0:
                new_seq.extend(seq[:edit_start])

            # Work on editable window
            i = edit_start
            while i < e_end:
                t = seq[i]

                # EXPAND: replace with two MASKs
                if t == expand_token:
                    new_seq.append(mask_token)
                    new_seq.append(mask_token)
                    i += 1
                    continue

                # DELETE: drop left real neighbor + the DELETE itself
                if t == delete_token:
                    # # find a real (non-sentinel) left neighbor inside the editable window *or* in prefix
                    # j = len(new_seq) - 1
                    # while j >= 0 and new_seq[j] in SENTINELS:
                    #     j -= 1
                    # if j >= 0:
                    #     new_seq.pop(j)   # remove the real token
                    # # skip the DELETE itself by not appending it
                    i += 1
                    continue

                # Otherwise keep the token
                new_seq.append(t)
                i += 1

            # Copy tail (non-editable)
            if e_end < len(seq):
                new_seq.extend(seq[e_end:])

            # truncate then pad with MASK so new slots are fillable next steps
            new_seq = new_seq[:max_len]
            padded = [mask_token] * max_len
            upto = min(len(new_seq), max_len)
            padded[:upto] = new_seq[:upto]
            out.append(torch.tensor(padded, device=x.device, dtype=x.dtype))

        return torch.stack(out, dim=0)

    def generate_sample(self, xt, text_ids, codes_ref, nsf: int, codes_ref_size: int):
        num_steps = nsf
        dt = 1.0 / num_steps
        x1_temp = 1.0
        gamma = 2.5
        mask_token_id = self.config.datasets.audio_mask_token
        S = self.config.datasets.audio_vocab_size + self.config.model.add_token
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