GPU_ID=5

CUDA_VISIBLE_DEVICES=$GPU_ID python3 /raid/aluno_alef/DFM-TTS-2/src/main.py \
    -c="/raid/aluno_alef/DFM-TTS-2/config/scaling_ablations_emilia/en-eos_as_pad-gpt2-emilia_yodas-mask-ce-poly-variable_window-c_coupling-pfg-1k.yaml"

CUDA_VISIBLE_DEVICES=$GPU_ID python3 /raid/aluno_alef/DFM-TTS-2/src/main.py \
    -c="/raid/aluno_alef/DFM-TTS-2/config/scaling_ablations_emilia/en-eos_as_pad-gpt2-emilia_yodas-mask-ce-poly-variable_window-c_coupling-pfg-10k.yaml"

CUDA_VISIBLE_DEVICES=$GPU_ID python3 /raid/aluno_alef/DFM-TTS-2/src/main.py \
    -c="/raid/aluno_alef/DFM-TTS-2/config/scaling_ablations_emilia/en-eos_as_pad-gpt2-emilia_yodas-mask-ce-poly-variable_window-c_coupling-pfg-full.yaml"