CUDA_ID=4

CHECKPOINTS_LIST=( \
    # 500h
    "/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/onfwsi1j/checkpoints/epoch=67-step=200000-val/loss_epoch=4.154.ckpt" \
    # 1000h
    "/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/npcez79d/checkpoints/epoch=31-step=190000-val/loss_epoch=4.207.ckpt" \
    # 5000h
    "/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/pv5oqzx6/checkpoints/epoch=06-step=200000-val/loss_epoch=4.212.ckpt" \
)

OUTPUT_DIRS=(
    # 500h
    "/raid/aluno_alef/DFM-TTS-2/src/output_infer/onfwsi1j_pfg_remdm" \
    # 1000h
    "/raid/aluno_alef/DFM-TTS-2/src/output_infer/npcez79d_pfg_remdm" \
    # 5000h
    "/raid/aluno_alef/DFM-TTS-2/src/output_infer/pv5oqzx6_pfg_remdm" \
)

HOURS=(500 1000 5000)

for i in "${!CHECKPOINTS_LIST[@]}"; do
    CHECKPOINT="${CHECKPOINTS_LIST[$i]}"
    OUTPUT_DIR="${OUTPUT_DIRS[$i]}"
    HOUR="${HOURS[$i]}"

    echo "Running inference for checkpoint: $CHECKPOINT (trained on $HOUR hours)"
    echo "Output will be saved to: $OUTPUT_DIR"

    CUDA_VISIBLE_DEVICES=$CUDA_ID python /raid/aluno_alef/DFM-TTS-2/src/infer_librispeech_v6_pfg_tsr_remask.py \
        --config="/raid/aluno_alef/DFM-TTS-2/config/scaling_ablations_emilia/en-eos_as_pad-gpt2-emilia_yodas-mask-ce-poly-variable_window-c_coupling-pfg-full.yaml" \
        --checkpoint="$CHECKPOINT" \
        --metadata_csv="/raid/aluno_alef/DATASETS/LibriSpeech-test-clean-filtered.csv" \
        --output_dir="$OUTPUT_DIR" \
        --use_oracle_length \
        --oracle_add_eos \
        --nsf=2,4,8,16,32,64,128 \
        --use_pfg \
        --gamma=1.5 \
        --use_remdm \
        --remdm_eta_rescale 0.5 \
        --remdm_eta_cap 0.5 \
        --remdm_tswitch 0.0
done