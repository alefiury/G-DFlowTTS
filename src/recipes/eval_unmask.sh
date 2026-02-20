CUDA_VISIBLE_DEVICES=4 python eval_unmask_bias.py \
    --config="/raid/aluno_alef/DFM-TTS-2/config/emilia/en-eos_as_pad-gpt2-emilia_yodas-mask-ce-poly-variable_window-c_coupling-pfg.yaml" \
    --checkpoint="/raid/aluno_alef/DFM-TTS-2/src/DFM-TTS/yrwxs6l7/checkpoints/epoch=01-step=710000-val/loss_epoch=3.927.ckpt" \
    --metadata_csv="/raid/aluno_alef/DATASETS/LibriSpeech-test-clean-filtered.csv" \
    --output_dir="output_infer/eval_unmask2-better_pfg_tsr-yrwxs6l7-en-eos_as_pad-gpt2-emilia_yodas-mask-ce-poly-variable_window-c_coupling-pfg" \
    --max_rows=100 \
    --use_oracle_length \
    --nsf=128 \
    --use_pfg \
    --gamma=2.5