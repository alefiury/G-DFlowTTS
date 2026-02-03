CUDA_VISIBLE_DEVICES=0 python infer_librispeech_v4_better_pfg_tsr.py \
    --config="/raid/aluno_alef/DFM-TTS-2/config/emilia/en-eos_as_pad-gpt2-emilia_yodas-mask-ce-poly-variable_window-c_coupling-pfg.yaml" \
    --checkpoint="/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/yrwxs6l7/checkpoints/epoch=01-step=710000-val/loss_epoch=3.927.ckpt" \
    --metadata_csv="/raid/aluno_alef/DATASETS/LibriSpeech-test-clean-filtered.csv" \
    --output_dir="output_infer/better_pfg_tsr-yrwxs6l7-en-eos_as_pad-gpt2-emilia_yodas-mask-ce-poly-variable_window-c_coupling-pfg" \
    --use_oracle_length \
    --nsf=4,8,16,32,128,256 \
    --use_pfg \
    --gamma=2.5 \