GPU_ID=7
CONFIG_PATH="/hadatasets/alef.ferreira/DFM-TTS-2/config/default_offline.yaml"
CHECKPOINT_PATH="/hadatasets/alef.ferreira/SLURM/DFM-TTS/wkfklcm8/checkpoints/last.ckpt"

CUDA_VISIBLE_DEVICES=$GPU_ID python3 inference.py \
    -c=$CONFIG_PATH \
    -pc=$CHECKPOINT_PATH