#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=7

CONFIG="/raid/aluno_alef/DFM-TTS-2/config/emilia/en-eos_as_pad-gpt2-emilia_yodas-mask-ce-poly-variable_window-c_coupling-pfg.yaml"
CKPT="/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/yrwxs6l7/checkpoints/epoch=01-step=880000-val/loss_epoch=3.937.ckpt"
META="/raid/aluno_alef/DATASETS/LibriSpeech-dev-clean-filtered.csv"
BASE_OUT="/raid/aluno_alef/DFM-TTS-2/src/output_infer/yrwxs6l7_grid_remdm_4gpu_full-dev_clean"

NSF="2,4,8,16,32,64,128"
GAMMA=1.5

TSWITCHES=0.7
ETA_RESCALES=0.2
ETA_CAPS=0.5

g_tag="${GAMMA/./p}"
OUT_G="${BASE_OUT}/gamma_${g_tag}"
mkdir -p "${OUT_G}"

OUT_DIR="${OUT_G}/remdm/ts_${ts_tag}/er_${er_tag}/cap_${cap_tag}"

python /raid/aluno_alef/DFM-TTS-2/src/infer_librispeech_v6_pfg_tsr_remask.py \
    --config="${CONFIG}" \
    --checkpoint="${CKPT}" \
    --metadata_csv="${META}" \
    --output_dir="${OUT_DIR}" \
    --gpu=0 \
    --use_oracle_length \
    --oracle_add_eos \
    --nsf="${NSF}" \
    --use_pfg \
    --gamma="${GAMMA}" \
    --use_remdm \
    --remdm_eta_rescale "${ETA_RESCALES}" \
    --remdm_eta_cap "${ETA_CAPS}" \
    --remdm_tswitch "${TSWITCHES}" \