import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn, Tensor

try:
    from flash_attn import flash_attn_func
    _flash_attn_available = True
except Exception:
    _flash_attn_available = False

from modules import rotary


# -------------------------
# helpers
# -------------------------
def bias_dropout_add_scale(
    x: Tensor,
    scale: Tensor,
    residual: Optional[Tensor],
    prob: float,
    training: bool,
) -> Tensor:
    if residual is None:
        residual = 0.0
    return residual + scale * F.dropout(x, p=prob, training=training)


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


class LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


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
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=time.device)
            / half
        )
        args = time[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, time: Tensor) -> Tensor:
        t_freq = self.timestep_embedding(time=time, dim=self.frequency_embedding_size)
        return self.mlp(t_freq)


def _make_sdpa_mask_from_valid(valid_mask: Tensor) -> Tensor:
    """
    valid_mask: (B, S) bool, True=valid keys (allowed)
    returns: (B, 1, 1, S) broadcastable for SDPA
    """
    if valid_mask.dim() != 2:
        raise ValueError(f"Expected (B,S) mask, got {tuple(valid_mask.shape)}")
    return valid_mask[:, None, None, :]


# -------------------------
# Attention modules
# -------------------------
class MultiHeadSelfAttention(nn.Module):
    """
    Self-attention on audio tokens.
    Applies RoPE on (q,k) using your rotary implementation.
    Respects audio_att_mask (True=valid).
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        proj_drop: float = 0.0,
        use_flash_if_available: bool = True,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim % 2 == 0, "RoPE requires even head_dim"
        self.use_flash = use_flash_if_available and _flash_attn_available

        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: Tensor,  # (B, Sa, D)
        audio_rotary_cos_sin: Optional[Tuple[Tensor, Tensor]],
        audio_att_mask: Optional[Tensor] = None,  # (B, Sa) bool True=valid
    ) -> Tensor:
        B, Sa, D = x.shape
        H, Hd = self.num_heads, self.head_dim

        q = self.q_linear(x).view(B, Sa, H, Hd)
        k = self.k_linear(x).view(B, Sa, H, Hd)
        v = self.v_linear(x).view(B, Sa, H, Hd)

        if audio_rotary_cos_sin is not None:
            with torch.amp.autocast("cuda", enabled=False):
                cos, sin = audio_rotary_cos_sin
                orig_dtype = q.dtype
                q = rotary.apply_rotary_emb_torch(q.float(), cos.float(), sin.float()).to(orig_dtype)
                k = rotary.apply_rotary_emb_torch(k.float(), cos.float(), sin.float()).to(orig_dtype)

        # FlashAttention only if no mask (masking breaks this simple path)
        if self.use_flash and audio_att_mask is None and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
            out = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)  # (B,Sa,H,Hd)
            out = rearrange(out, "b s h d -> b s (h d)")
        else:
            q_, k_, v_ = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

            attn_mask = None
            if audio_att_mask is not None:
                attn_mask = _make_sdpa_mask_from_valid(audio_att_mask.to(device=x.device, dtype=torch.bool))

            out = F.scaled_dot_product_attention(
                q_, k_, v_,
                attn_mask=attn_mask,
                dropout_p=0.0,
            )
            out = rearrange(out, "b h s d -> b s (h d)")

        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class MultiHeadCrossAttention(nn.Module):
    """
    Cross-attention:
      Q from audio, K/V from text.

    Text RoPE:
      Apply RoPE to text K only (encodes text order without a max-length table).
      Text is NOT padded to audio length. Only tokenizer batch padding exists and is masked.

    Mask convention:
      text_att_mask: (B, St) bool True=valid
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        proj_drop: float = 0.0,
        use_flash_if_available: bool = True,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim % 2 == 0, "RoPE requires even head_dim"
        self.use_flash = use_flash_if_available and _flash_attn_available

        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: Tensor,  # (B, Sa, D)
        text: Tensor,  # (B, St, D)
        text_att_mask: Optional[Tensor],  # (B, St) bool True=valid
        text_rotary_cos_sin: Optional[Tuple[Tensor, Tensor]],
    ) -> Tensor:
        B, Sa, D = x.shape
        St = text.shape[1]
        H, Hd = self.num_heads, self.head_dim

        q = self.q_linear(x).view(B, Sa, H, Hd)
        k = self.k_linear(text).view(B, St, H, Hd)
        v = self.v_linear(text).view(B, St, H, Hd)

        # --- TEXT RoPE on keys only ---
        if text_rotary_cos_sin is not None:
            with torch.amp.autocast("cuda", enabled=False):
                cos, sin = text_rotary_cos_sin
                orig_dtype = k.dtype
                k = rotary.apply_rotary_emb_torch(k.float(), cos.float(), sin.float()).to(orig_dtype)

        # FlashAttention only if no mask
        if self.use_flash and text_att_mask is None and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
            out = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)  # (B,Sa,H,Hd)
            out = rearrange(out, "b s h d -> b s (h d)")
        else:
            q_, k_, v_ = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

            attn_mask = None
            if text_att_mask is not None:
                attn_mask = _make_sdpa_mask_from_valid(text_att_mask.to(device=x.device, dtype=torch.bool))

            out = F.scaled_dot_product_attention(
                q_, k_, v_,
                attn_mask=attn_mask,
                dropout_p=0.0,
            )
            out = rearrange(out, "b h s d -> b s (h d)")

        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# -------------------------
# DiT block
# -------------------------
class DDiTBlockCross(nn.Module):
    """
    DiT-style adaLN-Zero block:
      SelfAttn(audio) -> CrossAttn(audio<-text) -> MLP
    Separate modulation per sublayer (9*dim): (shift,scale,gate) x3.
    """
    def __init__(
        self,
        dim: int,
        n_heads: int,
        cond_dim: int,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert dim % n_heads == 0
        self.dropout = dropout

        self.norm_sa = LayerNorm(dim)
        self.self_attn = MultiHeadSelfAttention(dim, n_heads, proj_drop=0.0)

        self.norm_ca = LayerNorm(dim)
        self.cross_attn = MultiHeadCrossAttention(dim, n_heads, proj_drop=0.0)

        self.norm_mlp = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )

        self.adaLN_modulation = nn.Linear(cond_dim, 9 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(
        self,
        x: Tensor,  # (B, Sa, D)
        c: Tensor,  # (B, cond_dim)
        audio_rotary_cos_sin: Optional[Tuple[Tensor, Tensor]],
        audio_att_mask: Optional[Tensor],  # (B, Sa) bool True=valid
        text_emb: Tensor,  # (B, St, D)
        text_att_mask: Optional[Tensor],  # (B, St) bool True=valid
        text_rotary_cos_sin: Optional[Tuple[Tensor, Tensor]],
    ) -> Tensor:
        (shift_sa, scale_sa, gate_sa,
         shift_ca, scale_ca, gate_ca,
         shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(9, dim=2)

        # --- self-attn ---
        x_in = modulate(self.norm_sa(x), shift=shift_sa, scale=scale_sa)
        sa_out = self.self_attn(x_in, audio_rotary_cos_sin=audio_rotary_cos_sin, audio_att_mask=audio_att_mask)
        x = bias_dropout_add_scale(sa_out, gate_sa, x, self.dropout, self.training)

        # --- cross-attn ---
        x_in = modulate(self.norm_ca(x), shift=shift_ca, scale=scale_ca)
        ca_out = self.cross_attn(
            x_in,
            text=text_emb,
            text_att_mask=text_att_mask,
            text_rotary_cos_sin=text_rotary_cos_sin,
        )
        x = bias_dropout_add_scale(ca_out, gate_ca, x, self.dropout, self.training)

        # --- mlp ---
        x_in = modulate(self.norm_mlp(x), shift=shift_mlp, scale=scale_mlp)
        mlp_out = self.mlp(x_in)
        x = bias_dropout_add_scale(mlp_out, gate_mlp, x, self.dropout, self.training)

        return x


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate(self.norm_final(x), shift=shift, scale=scale)
        return self.linear(x)


# -------------------------
# Full model
# -------------------------
class TransformerCrossAttn(nn.Module):
    """
    Cross-attention DiT (time-only adaLN conditioning, text through cross-attn only).

    Text RoPE:
      - computed from text_emb length St
      - applied to text keys in cross-attention

    Masks:
      audio_att_mask: (B, Sa) bool True=valid
      text_att_mask : (B, St) bool True=valid
    """
    def __init__(
        self,
        audio_vocab_size: int,
        text_vocab_size: int,
        hidden_size: int,
        cond_dim: int,
        n_heads: int,
        dropout: float,
        n_blocks: int,
        audio_add_token: int,          # mask + pad, etc.
        text_add_token: int,           # if you add extra symbols
        audio_pad_token: Optional[int] = None,
        text_pad_token: Optional[int] = None,
        text_filler_token: Optional[int] = None,  # used when drop_text=True
        mlp_ratio: int = 4,
        **kwargs,
    ):
        super().__init__()

        self.audio_vocab_size = int(audio_vocab_size)
        self.text_vocab_size = int(text_vocab_size)
        self.audio_add_token = int(audio_add_token)
        self.text_add_token = int(text_add_token)

        self.audio_pad_token = audio_pad_token
        self.text_pad_token = text_pad_token
        self.text_filler_token = int(text_filler_token) if text_filler_token is not None else 0

        true_audio_vocab = self.audio_vocab_size + self.audio_add_token
        true_text_vocab = self.text_vocab_size + self.text_add_token

        self.audio_embed = nn.Embedding(true_audio_vocab, hidden_size)
        self.text_embed = nn.Embedding(true_text_vocab, hidden_size)

        self.time_embedding = TimestepEmbedder(hidden_size=cond_dim)

        # RoPE: audio (self-attn) and text (cross-attn keys)
        self.audio_rotary = rotary.Rotary(dim=hidden_size // n_heads)
        self.text_rotary = rotary.Rotary(dim=hidden_size // n_heads)

        self.input_proj = nn.Linear(hidden_size, hidden_size)

        self.blocks = nn.ModuleList([
            DDiTBlockCross(
                dim=hidden_size,
                n_heads=n_heads,
                cond_dim=cond_dim,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(n_blocks)
        ])

        self.output_layer = DDitFinalLayer(
            hidden_size=hidden_size,
            out_channels=true_audio_vocab,
            cond_dim=cond_dim,
        )

    def forward(
        self,
        x_t: Tensor,                          # (B, Sa)
        text: Tensor,                         # (B, St)
        time: Tensor,                         # (B,)
        drop_text: bool = False,
        text_att_mask: Optional[Tensor] = None,   # (B, St) bool True=valid
        audio_att_mask: Optional[Tensor] = None,  # (B, Sa) bool True=valid
    ) -> Tensor:
        # --- audio ---
        x = self.audio_embed(x_t)         # (B, Sa, D)
        x = self.input_proj(x)

        B, Sa, _ = x.shape
        if audio_att_mask is None:
            audio_att_mask = torch.ones((B, Sa), device=x.device, dtype=torch.bool)
        else:
            audio_att_mask = audio_att_mask.to(device=x.device, dtype=torch.bool)

        # --- text (no padding to Sa) ---
        text = text.to(device=x.device)
        if text_att_mask is None:
            if self.text_pad_token is not None:
                text_att_mask = (text != self.text_pad_token)
            else:
                text_att_mask = torch.ones_like(text, dtype=torch.bool)
        text_att_mask = text_att_mask.to(device=x.device, dtype=torch.bool)

        if drop_text:
            # unconditional: replace ids with filler token, mark all as valid
            text = torch.full_like(text, fill_value=self.text_filler_token)
            text_att_mask = torch.ones_like(text_att_mask, dtype=torch.bool)

        text_emb = self.text_embed(text)  # (B, St, D)

        # --- time conditioning (time-only, per your preference) ---
        c = F.silu(self.time_embedding(time=time.to(device=x.device)))  # (B, cond_dim)

        # --- RoPE caches based on seq_len ---
        audio_rotary_cos_sin = self.audio_rotary(x=x)        # uses Sa
        text_rotary_cos_sin = self.text_rotary(x=text_emb)   # uses St

        # --- blocks ---
        for blk in self.blocks:
            x = blk(
                x=x,
                c=c,
                audio_rotary_cos_sin=audio_rotary_cos_sin,
                audio_att_mask=audio_att_mask,
                text_emb=text_emb,
                text_att_mask=text_att_mask,
                text_rotary_cos_sin=text_rotary_cos_sin,
            )

        # --- logits ---
        return self.output_layer(x=x, c=c)