#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=6

CONFIG="/raid/aluno_alef/DFM-TTS-2/config/emilia/en-eos_as_pad-gpt2-emilia_yodas-mask-ce-poly-variable_window-c_coupling-pfg.yaml"
CKPT="/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/yrwxs6l7/checkpoints/epoch=01-step=880000-val/loss_epoch=3.937.ckpt"
META="/raid/aluno_alef/DATASETS/LibriSpeech-test-clean-filtered.csv"
BASE_OUT="/raid/aluno_alef/DFM-TTS-2/src/output_infer/yrwxs6l7_grid_remdm_4gpu_full"

NSF="2,4,8,16,32,64,128"
GAMMAS=(1.5)

TSWITCHES=(0.0 0.5 0.7)
ETA_RESCALES=(0.2 0.3 0.5)
ETA_CAPS=(0.3 0.5)

PART_MOD=4
PART_REM=2

for g in "${GAMMAS[@]}"; do
  g_tag="${g/./p}"
  OUT_G="${BASE_OUT}/gamma_${g_tag}"
  mkdir -p "${OUT_G}"

  idx=0
  for ts in "${TSWITCHES[@]}"; do
    ts_tag="${ts/./p}"
    for er in "${ETA_RESCALES[@]}"; do
      er_tag="${er/./p}"
      for cap in "${ETA_CAPS[@]}"; do
        cap_tag="${cap/./p}"

        if (( idx % PART_MOD != PART_REM )); then
          idx=$((idx+1))
          continue
        fi

        OUT_DIR="${OUT_G}/remdm/ts_${ts_tag}/er_${er_tag}/cap_${cap_tag}"
        mkdir -p "${OUT_DIR}"

        echo "=== [GPU2 PART2] idx=${idx} gamma=${g} ts=${ts} er=${er} cap=${cap} -> ${OUT_DIR} ==="

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
          --gamma="${g}" \
          --use_remdm \
          --remdm_eta_rescale "${er}" \
          --remdm_eta_cap "${cap}" \
          --remdm_tswitch "${ts}" \
          2>&1 | tee "${OUT_DIR}/run.log"

        idx=$((idx+1))
      done
    done
  done
done
