CUDA_ID=5

CHECKPOINTS_LIST=( \
    # 10000h
    "/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/1lci8bnq/checkpoints/epoch=03-step=200000-val/loss_epoch=4.205.ckpt" \
    # 50000h
    "/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/fye3kna4/checkpoints/epoch=00-step=190000-val/loss_epoch=4.225.ckpt" \
    # full dataset
    "/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/11li9rse/checkpoints/epoch=00-step=180000-val/loss_epoch=4.413.ckpt" \
)

OUTPUT_DIRS=(
    # 10000h
    "/raid/aluno_alef/DFM-TTS-2/src/output_infer/1lci8bnq_pfg_remdm" \
    # 50000h
    "/raid/aluno_alef/DFM-TTS-2/src/output_infer/fye3kna4_pfg_remdm" \
    # full dataset
    "/raid/aluno_alef/DFM-TTS-2/src/output_infer/11li9rse_pfg_remdm" \
)

HOURS=(10000 50000 60000)

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