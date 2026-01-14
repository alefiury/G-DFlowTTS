CUDA_VISIBLE_DEVICES=7 python infer_librispeech_v4.py \
    --config="/raid/aluno_alef/DFM-TTS-2/config/en-eos_as_pad-gpt2-emilia_yodas-mask-kl-poly-variable_window.yaml" \
    --checkpoint="/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/i14puriu/checkpoints/epoch=00-step=920000-val/loss_epoch=1.697.ckpt" \
    --metadata_csv="/raid/aluno_alef/DATASETS/LibriSpeech-test-clean-filtered.csv" \
    --output_dir="output_infer/i14puriu-en-eos_as_pad-gpt2-emilia_yodas-mask-kl-poly-variable_window" \
    --use_oracle_length \
    --oracle_add_eos \
    --nsf=128 \