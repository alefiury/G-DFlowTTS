#!/usr/bin/env bash
set -euo pipefail

CONFIG="/raid/aluno_alef/DFM-TTS-2/config/emilia/en-eos_as_pad-gpt2-emilia_yodas-mask-ce-poly-variable_window-c_coupling-pfg.yaml"
CKPT="/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/yrwxs6l7/checkpoints/epoch=01-step=880000-val/loss_epoch=3.937.ckpt"
META="/raid/aluno_alef/DATASETS/LibriSpeech-test-clean-filtered.csv"

# Base output folder for the whole grid
BASE_OUT="output_infer/yrwxs6l7_tsr_grid"

NSF="2,4,8,16,32,64,128"
GAMMAS=(1.5)

# Grid to sweep (edit freely)
KS=(0.5 2.0 3.0)
SIGMAS=(1.25 2.0)

for g in "${GAMMAS[@]}"; do
  g_tag="${g/./p}"  # 1.5 -> 1p5

  for k in "${KS[@]}"; do
    k_tag="k${k}"

    for s in "${SIGMAS[@]}"; do
      s_tag="${s/./p}"  # 0.5 -> 0p5

      OUT_DIR="${BASE_OUT}/gamma_${g_tag}/${k_tag}/sigma_${s_tag}"

      echo "=== Running gamma=${g}, k=${k}, sigma=${s} -> ${OUT_DIR} ==="
      CUDA_VISIBLE_DEVICES=5 python infer_librispeech_v4-new_tsr.py \
        --config="${CONFIG}" \
        --checkpoint="${CKPT}" \
        --metadata_csv="${META}" \
        --output_dir="${OUT_DIR}" \
        --use_oracle_length \
        --oracle_add_eos \
        --nsf="${NSF}" \
        --use_pfg \
        --gamma="${g}" \
        --use_tsr \
        --tsr_k="${k}" \
        --tsr_sigma="${s}"
    done
  done
done
