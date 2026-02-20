#!/usr/bin/env bash
set -euo pipefail

CONFIG="/raid/aluno_alef/DFM-TTS-2/config/emilia/en-eos_as_pad-gpt2-emilia_yodas-mask-ce-poly-variable_window-pfg.yaml"
CKPT="/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/bzajs3jy/checkpoints/epoch=01-step=910000-val/loss_epoch=3.853.ckpt"
META="/raid/aluno_alef/DATASETS/LibriSpeech-test-clean-filtered.csv"
BASE_OUT="output_infer/bzajs3jy"

NSF="2,4,8,16,32,64,128"
GAMMAS=(0.5 1 1.5)

for g in "${GAMMAS[@]}"; do
  g_tag="${g/./p}"                 # 0.5 -> 0p5 (evita '.' no nome da pasta)
  OUT_DIR="${BASE_OUT}/gamma_${g_tag}"

  echo "=== Running gamma=${g} -> ${OUT_DIR} ==="
  python infer_librispeech_v4.py \
    --config="${CONFIG}" \
    --checkpoint="${CKPT}" \
    --metadata_csv="${META}" \
    --output_dir="${OUT_DIR}" \
    --use_oracle_length \
    --oracle_add_eos \
    --nsf="${NSF}" \
    --use_pfg \
    --gamma="${g}"
done
