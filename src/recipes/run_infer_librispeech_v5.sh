CUDA_VISIBLE_DEVICES=0 python infer_librispeech_v5.py \
    --config="/raid/aluno_alef/DFM-TTS-2/config/libritts_r/en-eos_as_pad-gpt2-mask-ce-poly-variable_window-pfg.yaml" \
    --checkpoint="/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/cw5568eu/checkpoints/epoch=20-step=460000-val/loss_epoch=1.630.ckpt" \
    --metadata_csv="/raid/aluno_alef/DATASETS/LibriSpeech-test-clean-filtered.csv" \
    --output_dir="output_infer/cw5568eu-en-eos_as_pad-gpt2-mask-ce-poly-variable_window-pfg" \
    --use_oracle_length \
    --nsf=4,8,16,32,128,256,512,1024,2048