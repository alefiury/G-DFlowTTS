# Discrete Flow Matching TTS

## TAGARELA v2 streaming Parquet fine-tuning

Use `config/pt-tagarela-v2-streaming-finetune.yaml`. The streamed Parquet rows
use `audio` as raw waveform input and `stt_parakeet` as text. NeuCodec targets
are extracted on the training GPU per utterance.

Fine-tune weights with a fresh optimizer/scheduler:

```bash
python3 src/main.py \
  -c config/pt-tagarela-v2-streaming-finetune.yaml \
  -pc /path/to/pretrained.ckpt \
  -g 0
```

Start from scratch by omitting `-pc`. Resume an interrupted run including
optimizer/scheduler/global-step state with:

```bash
python3 src/main.py \
  -c config/pt-tagarela-v2-streaming-finetune.yaml \
  -pc /path/to/last.ckpt \
  --continue-training \
  -g 0
```

