python infer_librispeech_v5.py \
    --config="/raid/aluno_alef/DFM-TTS-2/config/emilia/en-eos_as_pad-gpt2-emilia_yodas-mask-ce-poly-variable_window.yaml" \
    --checkpoint="/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/jggxs128/checkpoints/epoch=01-step=800000-val/loss_epoch=3.921.ckpt" \
    --metadata_csv="/raid/aluno_alef/DATASETS/LibriSpeech-test-clean-filtered.csv" \
    --output_dir="output_infer/jggxs128" \
    --use_oracle_length \
    --oracle_add_eos \
    --nsf=2,4,8,16,32,64,128