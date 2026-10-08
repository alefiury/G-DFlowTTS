"""G-DFlowTTS for 🤗 Transformers (``trust_remote_code=True``).

Self-contained port of ``modules/gdflowtts/model.py`` plus the CTMC sampler of
``utils/sampling.py``. Only ``torch`` and ``transformers`` are required;
``torchaudio`` is used only to resample reference audio that is not 16 kHz.
"""

import math
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer, PreTrainedModel

from .configuration_gdflowtts import GDFlowTTSConfig


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class LayerNorm(nn.Module):
    """Bias-free LayerNorm computed in float32."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        out = F.layer_norm(x.float(), [self.dim]) * self.weight.float()[None, None, :]
        return out.to(x.dtype)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(time: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=time.device) / half
        )
        args = time[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, time: Tensor) -> Tensor:
        t_freq = self.timestep_embedding(time, self.frequency_embedding_size)
        return self.mlp(t_freq.to(self.mlp[0].weight.dtype))


class DDiTBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, cond_dim: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.dropout = dropout

        self.norm1 = LayerNorm(dim)
        self.qw = nn.Linear(dim, dim, bias=False)
        self.kw = nn.Linear(dim, dim, bias=False)
        self.vw = nn.Linear(dim, dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, c: Tensor, att_mask: Tensor) -> Tensor:
        batch_size, seq_len = x.shape[0], x.shape[1]
        att_mask = att_mask[:, None, None, :].to(device=x.device, dtype=torch.bool)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
        )

        x_skip = x
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        q, k, v = (
            proj(h).view(batch_size, seq_len, self.n_heads, self.head_dim)
            for proj in (self.qw, self.kw, self.vw)
        )

        # Rotary embedding on q/k, computed in float32
        q = (q.float() * cos + rotate_half(q.float()) * sin).to(v.dtype)
        k = (k.float() * cos + rotate_half(k.float()) * sin).to(v.dtype)

        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=att_mask)
        h = h.transpose(1, 2).reshape(batch_size, seq_len, -1)

        x = x_skip + gate_msa * F.dropout(self.attn_out(h), p=self.dropout, training=self.training)
        h = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = x + gate_mlp * F.dropout(h, p=self.dropout, training=self.training)
        return x


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class GDFlowTTSPreTrainedModel(PreTrainedModel):
    config_class = GDFlowTTSConfig
    base_model_prefix = "gdflowtts"
    main_input_name = "x_t"
    _no_split_modules = ["DDiTBlock"]
    _supports_sdpa = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, LayerNorm):
            nn.init.ones_(module.weight)


class GDFlowTTSModel(GDFlowTTSPreTrainedModel):
    def __init__(self, config: GDFlowTTSConfig):
        super().__init__(config)
        self.total_audio_vocab_size = config.audio_vocab_size + config.audio_add_token
        total_text_vocab_size = config.text_vocab_size + config.text_add_token

        self.audio_embed = nn.Embedding(self.total_audio_vocab_size, config.hidden_size)
        self.text_embed = nn.Embedding(total_text_vocab_size, config.hidden_size)
        self.time_embedding = TimestepEmbedder(config.cond_dim, config.frequency_embedding_size)
        # Audio and text embeddings are concatenated, then projected back
        self.input_proj = nn.Linear(config.hidden_size * 2, config.hidden_size)

        self.blocks = nn.ModuleList(
            [
                DDiTBlock(config.hidden_size, config.n_heads, config.cond_dim, config.mlp_ratio, config.dropout)
                for _ in range(config.n_blocks)
            ]
        )
        self.output_layer = DDitFinalLayer(config.hidden_size, self.total_audio_vocab_size, config.cond_dim)

        # Lazily loaded helpers; kept out of the module tree so they are never saved
        self.__dict__["_codec"] = None
        self.__dict__["_feature_extractor"] = None
        self.__dict__["_tokenizer"] = None

        self.post_init()

    def _rotary_cos_sin(self, seq_len: int, device: torch.device):
        head_dim = self.config.hidden_size // self.config.n_heads
        inv_freq = 1.0 / (
            self.config.rotary_base
            ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
        )
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        # [1, seq_len, 1, head_dim] to broadcast over (batch, seq, heads, head_dim)
        return emb.cos()[None, :, None, :], emb.sin()[None, :, None, :]

    def forward(
        self,
        x_t: Tensor,
        text_ids: Tensor,
        time: Tensor,
        drop_text: bool = False,
        text_att_mask: Optional[Tensor] = None,
        audio_att_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Returns logits over the audio vocabulary, shape [B, T, audio_vocab_size + audio_add_token]."""
        seq_len = x_t.shape[1]
        audio_emb = self.audio_embed(x_t)

        if audio_att_mask is None:
            audio_att_mask = torch.ones(x_t.shape, device=x_t.device, dtype=torch.bool)

        # Text tokens are aligned position-wise with audio: truncate/pad to audio length
        text = text_ids[:, :seq_len]
        text = F.pad(text, (0, seq_len - text.shape[1]), value=self.config.text_filler_token)
        if drop_text:
            text = torch.full_like(text, self.config.text_filler_token)
        text_emb = self.text_embed(text)

        x = self.input_proj(torch.cat([audio_emb, text_emb], dim=-1))
        c = F.silu(self.time_embedding(time))
        cos, sin = self._rotary_cos_sin(seq_len, x.device)

        for block in self.blocks:
            x = block(x, cos, sin, c, audio_att_mask)

        return self.output_layer(x, c)

    # ------------------------------------------------------------------
    # Codec / tokenizer helpers
    # ------------------------------------------------------------------

    def load_codec(self, codec_name: Optional[str] = None):
        """Loads NeuCodec (native in Transformers) on the model's device."""
        if self._codec is None:
            name = codec_name or self.config.codec_name
            self.__dict__["_codec"] = AutoModel.from_pretrained(name).to(self.device).eval()
            self.__dict__["_feature_extractor"] = AutoFeatureExtractor.from_pretrained(name)
        return self._codec

    def load_tokenizer(self, tokenizer_name: Optional[str] = None):
        if self._tokenizer is None:
            # The tokenizer lives in this (already trusted) repo, whose config.json has an
            # auto_map; without trust_remote_code AutoTokenizer would prompt interactively.
            tok = AutoTokenizer.from_pretrained(tokenizer_name or self.config._name_or_path, trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            self.__dict__["_tokenizer"] = tok
        return self._tokenizer

    @torch.no_grad()
    def encode_audio(self, audio: Union[np.ndarray, Tensor], sampling_rate: int) -> Tensor:
        """Encodes a mono waveform into 1-D NeuCodec codes."""
        codec = self.load_codec()
        audio = torch.as_tensor(audio, dtype=torch.float32).cpu()
        if audio.ndim > 1:
            audio = audio.reshape(-1, audio.shape[-1]).mean(dim=0)
        target_sr = self.config.codec_input_sampling_rate
        if sampling_rate != target_sr:
            try:
                import torchaudio
            except ImportError as e:
                raise ValueError(
                    f"Reference audio must be {target_sr} Hz (got {sampling_rate}); "
                    "install torchaudio for automatic resampling."
                ) from e
            audio = torchaudio.functional.resample(audio, sampling_rate, target_sr)

        inputs = self._feature_extractor(
            audio=[audio.numpy()], sampling_rate=target_sr, return_tensors="pt"
        ).to(self.device, codec.dtype)
        return codec.encode(**inputs).audio_codes.long().reshape(-1)

    @torch.no_grad()
    def decode_audio(self, codes: Tensor) -> Tensor:
        """Decodes 1-D NeuCodec codes into a 24 kHz waveform."""
        codec = self.load_codec()
        return codec.decode(codes.to(self.device).long().view(1, 1, -1)).audio_values.reshape(-1).float().cpu()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _scheduler(self, t: float):
        """Polynomial convex scheduler: returns (alpha_t, d alpha_t / dt)."""
        n = float(self.config.scheduler_exponent)
        if t == 0.0 and n < 1.0:
            return 0.0, float("inf")
        return t**n, n * t ** (n - 1.0)

    @torch.no_grad()
    def sample_codes(
        self,
        text_ids: Tensor,
        ref_codes: Tensor,
        suffix_len: int,
        steps: int = 128,
        text_att_mask: Optional[Tensor] = None,
        x1_temp: float = 1.0,
        temp_schedule: str = "dfm36",
        remask_noise: float = 0.0,
        use_pfg: bool = False,
        gamma: float = 1.5,
        use_tsr: bool = False,
        tsr_k: float = 1.0,
        tsr_sigma: float = 0.1,
        use_sc_remask: bool = False,
        sc_remask_eta_rescale: float = 0.5,
        sc_remask_eta_cap: float = 0.5,
        sc_remask_tswitch: float = 0.0,
        sc_remask_use_conf: bool = False,
        sc_remask_conf_threshold: float = 0.35,
        sc_remask_beta: float = 2.0,
        sc_remask_strength: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Masked-source CTMC sampler (tau-leaping) with optional PFG, TSR and SC-ReMask.

        Returns the full sequence [1, len(ref_codes) + suffix_len], whose last
        position is pinned to EOS.
        """
        device = self.device
        S = self.total_audio_vocab_size
        mask_id = int(self.config.audio_mask_token)
        eos_id = int(self.config.audio_eos_token)
        eps = 1e-12

        if use_pfg and float(self.config.cond_drop_prob) <= 0.0:
            raise RuntimeError("use_pfg=True but this checkpoint was trained without text dropout.")

        ref_codes = ref_codes.to(device).long().reshape(-1)
        text_ids = text_ids.to(device)
        if text_att_mask is not None:
            text_att_mask = text_att_mask.to(device)
        prefix_len = int(ref_codes.numel())
        total_len = prefix_len + int(suffix_len)

        xt = torch.full((1, total_len), mask_id, device=device, dtype=torch.long)
        xt[:, :prefix_len] = ref_codes
        xt[0, -1] = eos_id
        audio_att_mask = torch.ones_like(xt, dtype=torch.bool)

        dt = 1.0 / max(1, int(steps))

        def logits_at(t: Tensor, drop_text: bool) -> Tensor:
            return self(
                x_t=xt,
                text_ids=text_ids,
                time=t,
                drop_text=drop_text,
                text_att_mask=text_att_mask,
                audio_att_mask=audio_att_mask,
            ).float()

        for k in range(int(steps)):
            t_lin = k * dt
            t = torch.full((1,), t_lin, device=device, dtype=torch.float32)
            alpha_t, dalpha_t = self._scheduler(t_lin)
            alpha_s, _ = self._scheduler(min(1.0, (k + 1) * dt))
            dalpha_t = max(dalpha_t, 1e-6)
            lam = dalpha_t / max(1.0 - alpha_t, 1e-6)

            if temp_schedule == "dfm36":
                temp = max(1e-3, float(x1_temp) * (1.0 - t_lin) ** 2)
            else:
                temp = max(1e-3, float(x1_temp))

            # Temporal Score Rescaling
            tsr_ratio = 1.0
            if use_tsr and float(tsr_k) != 1.0:
                a = min(max(alpha_t, 0.0), 1.0)
                num = (1.0 - a) + a * tsr_sigma**2
                den = (1.0 - a) + a * tsr_sigma**2 / float(tsr_k)
                tsr_ratio = max(num / max(den, eps), eps)

            def probs_from(logits: Tensor) -> Tensor:
                p = torch.softmax(logits * tsr_ratio / temp, dim=-1)
                p[..., mask_id] = 0.0
                return p / p.sum(dim=-1, keepdim=True).clamp_min(eps)

            probs_c = probs_from(logits_at(t, drop_text=False))
            probs_u = probs_from(logits_at(t, drop_text=True)) if use_pfg else None

            xt_is_mask = (xt == mask_id).unsqueeze(-1).float()
            base_r = (1.0 + float(remask_noise) * t_lin) * lam
            R_c = xt_is_mask * probs_c * base_r
            R_u = xt_is_mask * probs_u * base_r if use_pfg else None

            if remask_noise > 0.0:
                to_mask = (1.0 - xt_is_mask) * F.one_hot(torch.tensor(mask_id, device=device), S).float()
                R_c = R_c + to_mask * float(remask_noise)
                if R_u is not None:
                    R_u = R_u + to_mask * float(remask_noise)

            # Prediction-free guidance in log-rate space
            if use_pfg:
                R_mix = torch.exp(gamma * torch.log(R_c + 1e-9) + (1.0 - gamma) * torch.log(R_u + 1e-9))
            else:
                R_mix = R_c

            # SC-ReMask: schedule-constrained remasking (token -> MASK)
            if use_sc_remask:
                sigma_max = min(1.0, max(1.0 - alpha_s, 0.0) / max(alpha_t, eps))
                sigma = sc_remask_eta_rescale * min(sc_remask_eta_cap, sigma_max)
                sigma = min(max(sigma, 0.0), 1.0 - 1e-6)
                if t_lin < sc_remask_tswitch:
                    sigma = 0.0
                if sigma > 0.0:
                    r_base = -math.log(max(1e-9, 1.0 - sigma)) / max(1e-12, dt)
                    eligible = torch.ones_like(xt, dtype=torch.bool)
                    eligible[:, :prefix_len] = False
                    eligible[:, -1] = False
                    eligible = eligible & (xt != mask_id) & (xt != eos_id)

                    if sc_remask_use_conf:
                        if use_pfg:
                            logp = gamma * torch.log(probs_c + eps) + (1.0 - gamma) * torch.log(probs_u + eps)
                            probs_mix = torch.exp(logp)
                            probs_mix = probs_mix / probs_mix.sum(dim=-1, keepdim=True).clamp_min(eps)
                        else:
                            probs_mix = probs_c
                        p_cur = probs_mix.gather(-1, xt[..., None].clamp(0, S - 1)).squeeze(-1)
                        thr = float(sc_remask_conf_threshold)
                        w = (p_cur < thr).float() * ((thr - p_cur) / max(1e-9, thr)).clamp(0.0, 1.0) ** sc_remask_beta
                        w = (w * float(sc_remask_strength)).clamp(0.0, 1.0)
                    else:
                        w = torch.ones_like(xt, dtype=torch.float32)

                    add = torch.zeros_like(R_mix)
                    add[..., mask_id] = (r_base * w).to(R_mix.dtype)
                    R_mix = R_mix + add * eligible.unsqueeze(-1).to(R_mix.dtype)

            # CTMC tau-leap step
            R_off = R_mix.clone()
            R_off.scatter_(-1, xt[..., None], 0.0)
            hazard = R_off.sum(dim=-1)
            p_jump = 1.0 - torch.exp(-dt * hazard)
            do_jump = torch.rand(p_jump.shape, device=device, generator=generator) < p_jump
            do_jump[:, :prefix_len] = False
            do_jump[:, -1] = False

            if do_jump.any():
                q = R_off / hazard.clamp_min(1e-9).unsqueeze(-1)
                jump_idx = do_jump.view(-1).nonzero(as_tuple=False).squeeze(-1)
                q_jump = q.view(-1, S).index_select(0, jump_idx)
                q_jump = torch.nan_to_num(q_jump, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
                q_jump = q_jump / q_jump.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                sampled = torch.multinomial(q_jump, 1, generator=generator).squeeze(-1)
                xt.view(-1)[jump_idx] = sampled

            xt[0, -1] = eos_id
            xt[:, :prefix_len] = ref_codes

            # Early stop only when nothing can be remasked anymore
            if remask_noise <= 0.0 and not use_sc_remask and not (xt[:, prefix_len:] == mask_id).any():
                break

        return xt

    @torch.no_grad()
    def synthesize(
        self,
        text: str,
        ref_audio: Union[np.ndarray, Tensor],
        ref_text: str,
        ref_sampling_rate: int = 16_000,
        steps: int = 128,
        speed: float = 1.0,
        duration: Optional[int] = None,
        seed: Optional[int] = None,
        **sampling_kwargs,
    ) -> Tensor:
        """Zero-shot TTS: clones the voice of ``ref_audio`` (transcribed as ``ref_text``).

        Args:
            text: text to synthesize.
            ref_audio: mono reference waveform.
            ref_text: transcription of the reference audio.
            ref_sampling_rate: sampling rate of ``ref_audio``.
            steps: number of CTMC sampling steps.
            speed: speaking rate for the length heuristic (>1 is faster).
            duration: number of codec frames (50 Hz) to generate; overrides ``speed``.
            seed: optional random seed.
            **sampling_kwargs: forwarded to :meth:`sample_codes` (x1_temp, use_pfg, gamma, use_tsr, ...).

        Returns:
            1-D float tensor with the waveform at ``config.sampling_rate`` (24 kHz).
        """
        tokenizer = self.load_tokenizer()
        enc = tokenizer(ref_text + ". " + text, return_tensors="pt")
        ref_codes = self.encode_audio(ref_audio, ref_sampling_rate)

        if duration is None:
            duration = math.floor((ref_codes.numel() / max(1, len(ref_text))) * (len(text) / speed))
        # +1 for the EOS token pinned at the last position
        suffix_len = int(duration) + 1

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        xt = self.sample_codes(
            text_ids=enc["input_ids"],
            text_att_mask=enc["attention_mask"].bool(),
            ref_codes=ref_codes,
            suffix_len=suffix_len,
            steps=steps,
            generator=generator,
            **sampling_kwargs,
        )

        gen = xt[0, ref_codes.numel():]
        eos_pos = (gen == self.config.audio_eos_token).nonzero(as_tuple=False)
        if eos_pos.numel() > 0:
            gen = gen[: int(eos_pos[0])]
        gen = gen[gen != self.config.audio_mask_token]
        return self.decode_audio(gen)


__all__ = ["GDFlowTTSPreTrainedModel", "GDFlowTTSModel"]
