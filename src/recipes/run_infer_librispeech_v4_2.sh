CUDA_VISIBLE_DEVICES=7 python infer_librispeech_v4.py \
    --config="/raid/aluno_alef/DFM-TTS-2/config/en-eos_as_pad-gpt2-emilia_yodas-mask-kl-poly-variable_window.yaml" \
    --checkpoint="/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/cw5568eu/checkpoints/epoch=20-step=460000-val/loss_epoch=1.630.ckpt" \
    --metadata_csv="/raid/aluno_alef/DATASETS/LibriSpeech-test-clean-filtered.csv" \
    --output_dir="output_infer/cw5568eu-en-eos_as_pad-gpt2-mask-ce-poly-variable_window-pfg" \
    --use_oracle_length \
    --nsf=4,8,16,32,128,256 \
    # --use_pfg \
    # --gamma=2.5 \