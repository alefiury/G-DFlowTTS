CUDA_VISIBLE_DEVICES=7 python infer_librispeech_v4.py \
    --config="/raid/aluno_alef/DFM-TTS-2/config/en-eos_as_pad-gpt2-emilia_yodas-mask-kl-poly-variable_window.yaml" \
    --checkpoint="/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/zs5nx1pw/checkpoints/epoch=00-step=870000-val/loss_epoch=1.683.ckpt" \
    --metadata_csv="/raid/aluno_alef/DATASETS/LibriSpeech-test-clean-filtered.csv" \
    --output_dir="output_infer/zs5nx1pw-en-eos_as_pad-gpt2-pfg_0_1-emilia_yodas-mask-kl-poly-variable_window" \
    --use_oracle_length \
    --oracle_add_eos \
    --nsf=4,8,16,32,128,256 \
    --use_pfg \
    --gamma=2.0 \