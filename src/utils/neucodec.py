from typing import Optional

import torch
from torch import nn, Tensor
from transformers import AutoFeatureExtractor, AutoModel


class NeuCodec(nn.Module):
    """NeuCodec loaded through 🤗 Transformers, exposing the ``encode_code`` /
    ``decode_code`` API of the original ``neucodec`` package so call sites stay
    unchanged. Encodes 16 kHz audio into 50 Hz FSQ codes and decodes them to
    24 kHz audio.
    """

    input_sample_rate = 16_000
    sample_rate = 24_000

    def __init__(self, model: nn.Module, feature_extractor):
        super().__init__()
        self.model = model
        self.feature_extractor = feature_extractor

    @classmethod
    def from_pretrained(cls, model_id: str = "neuphonic/neucodec", revision: Optional[str] = None) -> "NeuCodec":
        model = AutoModel.from_pretrained(model_id, revision=revision)
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, revision=revision)
        return cls(model, feature_extractor).eval()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def encode_code(self, audio: Tensor) -> Tensor:
        """
        Args:
            audio: 16 kHz waveform, [B, 1, T], [1, T] or [T]

        Returns:
            fsq_codes: [B, 1, F] 50 Hz FSQ codes
        """
        audio = audio.detach().float().cpu()
        if audio.ndim == 1:
            audio = audio[None, None, :]
        elif audio.ndim == 2:
            audio = audio[:, None, :]
        if audio.ndim != 3 or audio.size(1) != 1:
            raise ValueError(f"NeuCodec input must be [B, 1, T], got {tuple(audio.shape)}")

        # The feature extractor expects 1-D waveforms and computes mel features on CPU
        inputs = self.feature_extractor(
            audio=[example[0].numpy() for example in audio],
            sampling_rate=self.input_sample_rate,
            return_tensors="pt",
        ).to(self.device, self.model.dtype)
        return self.model.encode(**inputs).audio_codes.long()

    @torch.no_grad()
    def decode_code(self, fsq_codes: Tensor) -> Tensor:
        """
        Args:
            fsq_codes: [B, 1, F] 50 Hz FSQ codes

        Returns:
            recon: [B, 1, T] reconstructed 24 kHz audio
        """
        return self.model.decode(fsq_codes.to(self.device).long()).audio_values
