import os
import json
import argparse

import wandb
import torch
import torchaudio
from omegaconf import OmegaConf
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from modules.wrappers.pl_wrapper import DFMTTSWrapper

torch.autograd.set_detect_anomaly(True) # for debugging

wandb.finish()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_path",
        required=True,
        type=str,
        help="YAML file with configurations"
    )
    parser.add_argument(
        "-g",
        "--gpu",
        default=0,
        required=False,
        type=int
    )
    parser.add_argument(
        "-gpus",
        "--gpus",
        required=False,
        type=json.loads,
        default=None
    )
    parser.add_argument(
        "-ck",
        "--checkpoint-dir",
        required=False,
        type=str,
        default="../checkpoints/DFMTTS"
    )
    parser.add_argument(
        "-pc",
        "--pretrained-checkpoint",
        required=False,
        type=str,
        default=None
    )
    parser.add_argument(
        "--continue-training",
        action="store_true",
        help="Whether to continue training from the latest checkpoint in the checkpoint directory"
    )

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    tags = []
    tags += config.tags  # add tags defined for experiments
    exp_title = config.title

    wandb.init(
        project=config.wandb_project_name,
        name=exp_title,
        tags=tags,
        entity=config.wandb_entity,
        config=OmegaConf.to_container(config, resolve=True)
    )

    logger = WandbLogger(
        project=config.wandb_project_name,
        name=exp_title,
        tags=tags,
        entity=config.wandb_entity,
        config=OmegaConf.to_container(config, resolve=True)
    )

    if config.test.get("log_audio_ref", False):
        wav, sr = torchaudio.load(config.test.audio_ref_path)
        wandb.log(
            {
                "audio_ref": wandb.Audio(
                    wav.squeeze(0).numpy(),
                    sample_rate=sr,
                    caption=config.test.text_ref,
                ),
            }
        )

    no_streaming_validation = (
        config.datasets.type == "hf_streaming_text_tokenizer"
        and not config.datasets.get("val_metadata", "")
    ) or (
        config.datasets.type == "hf_streaming_codes"
        and int(config.datasets.get("val_num_samples", 0)) <= 0
    )

    checkpoint_config = OmegaConf.to_container(
        config["model_checkpoint"], resolve=True
    )
    checkpoint_config.pop("dirpath", None)

    if no_streaming_validation:
        monitor = checkpoint_config.get("monitor")
        if monitor and str(monitor).startswith("val/"):
            print(
                "No validation dataset configured: disabling validation-metric "
                "checkpoint monitoring and saving periodic checkpoints instead."
            )
            checkpoint_config.pop("monitor", None)
            checkpoint_config.pop("mode", None)
            checkpoint_config["save_top_k"] = 1

        filename = checkpoint_config.get("filename", "")
        if "val/" in filename:
            checkpoint_config["filename"] = "{epoch:02d}-{step:08d}"

    callbacks = [
        ModelCheckpoint(**checkpoint_config),
        LearningRateMonitor("step"),
    ]

    if args.pretrained_checkpoint is not None and not args.continue_training:
        print("*" * 100)
        print("Fine-tuning weights from checkpoint:", args.pretrained_checkpoint)

        # Fine-tuning != resume: load only model weights, then let Trainer create
        # a fresh optimizer/scheduler and start step counting from zero.
        model = DFMTTSWrapper(config=config)
        checkpoint = torch.load(
            args.pretrained_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict,
            strict=False,
        )
        print("Loaded pretrained model weights.")
        codec_missing = [
            key for key in missing_keys if key.startswith("audio_codec.")
        ]
        other_missing = [
            key for key in missing_keys if not key.startswith("audio_codec.")
        ]
        if codec_missing:
            print(
                f"Ignored {len(codec_missing)} missing frozen audio_codec keys "
                "(they are intentionally not stored in TTS checkpoints)."
            )
        if other_missing:
            print("WARNING - other missing model keys:", other_missing)
        if unexpected_keys:
            print("WARNING - unexpected checkpoint keys:", unexpected_keys)
    else:
        model = DFMTTSWrapper(config=config)

    print(model)

    trainer_config = OmegaConf.to_container(config["trainer"], resolve=True)
    if no_streaming_validation:
        # Be explicit: do not run sanity checks or validation loops when there
        # is no validation stream. No training examples are held out.
        trainer_config["limit_val_batches"] = 0
        trainer_config["num_sanity_val_steps"] = 0

    if args.gpus is not None:
        trainer = Trainer(
            **trainer_config,
            logger=logger,
            callbacks=callbacks,
            devices=args.gpus,
            strategy=DDPStrategy(process_group_backend="gloo", find_unused_parameters=True),
            default_root_dir=os.path.join(args.checkpoint_dir, config["title"])
        )
    else:
        trainer = Trainer(
            **trainer_config,
            logger=logger,
            callbacks=callbacks,
            devices=[args.gpu],
            default_root_dir=os.path.join(args.checkpoint_dir, config["title"])
        )

    if args.continue_training:
        print("*"*100)
        print("Continuing training from the latest checkpoint in:", args.pretrained_checkpoint)
        latest_checkpoint = args.pretrained_checkpoint
        trainer.fit(model, ckpt_path=latest_checkpoint)
    else:
        trainer.fit(model)


if __name__ == "__main__":
    main()
