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

from modules.pl_wrapper import DFMTTSWrapper

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

    config["model_checkpoint"].pop("dirpath")

    callbacks = [
        ModelCheckpoint(**config["model_checkpoint"]),
        LearningRateMonitor("step"),
    ]

    if args.pretrained_checkpoint is not None and not args.continue_training:
        print("*"*100)
        print("Fine-tuning from checkpoint:", args.pretrained_checkpoint)
        model = DFMTTSWrapper.load_from_checkpoint(args.pretrained_checkpoint, config=config)
        print("Loaded model from checkpoint:", args.pretrained_checkpoint)
    else:
        model = DFMTTSWrapper(config=config)

    print(model)

    if args.gpus is not None:
        trainer = Trainer(
            **config["trainer"],
            logger=logger,
            callbacks=callbacks,
            devices=args.gpus,
            strategy=DDPStrategy(process_group_backend="gloo", find_unused_parameters=True),
            default_root_dir=os.path.join(args.checkpoint_dir, config["title"])
        )
    else:
        trainer = Trainer(
            **config["trainer"],
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
