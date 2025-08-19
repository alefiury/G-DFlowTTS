import math
from typing import Optional

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


def bias_dropout_add_scale(
    x: Tensor,
    scale: Tensor,
    residual: Optional[Tensor],
    prob: float,
    training: bool
) -> Tensor:
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
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=time.device)
        args = time[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, time: Tensor) -> Tensor:
        t_freq = self.timestep_embedding(time=time, dim=self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class MultiHeadSelfAttention(nn.Module):
    """
    Self-attention with rotary on (q,k). Uses FlashAttention if available and no attn_mask;
    otherwise falls back to PyTorch SDPA. Returns projected output.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.attn_drop = attn_drop

        self.q_linear  = nn.Linear(d_model, d_model, bias=False)
        self.k_linear  = nn.Linear(d_model, d_model, bias=False)
        self.v_linear  = nn.Linear(d_model, d_model, bias=False)
        self.proj      = nn.Linear(d_model, d_model, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: Tensor, # (B, S, D)
        rotary_cos_sin: Optional[Tensor], # tuple(cos, sin) from rotary for audio stream
        attn_mask: Optional[Tensor] = None # if needed in future; None by default
    ) -> Tensor:
        B, S, D = x.shape
        H, Hd   = self.num_heads, self.head_dim

        # Projections
        q = self.q_linear(x).view(B, S, H, Hd)
        k = self.k_linear(x).view(B, S, H, Hd)
        v = self.v_linear(x).view(B, S, H, Hd)

        # Rotary on q,k
        if rotary_cos_sin is not None:
            with torch.amp.autocast("cuda", enabled=False):
                cos, sin = rotary_cos_sin
                orig_dtype = q.dtype
                q = rotary.apply_rotary_emb_torch(q.float(), cos.float(), sin.float()).to(orig_dtype)
                k = rotary.apply_rotary_emb_torch(k.float(), cos.float(), sin.float()).to(orig_dtype)

        # SDPA expects (B,H,S,D)
        q_, k_, v_ = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q_, k_, v_,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop if self.training else 0.0
        ) # (B,H,S,Hd)
        out = rearrange(out, "b h s d -> b s (h d)")

        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class MultiHeadCrossAttention(nn.Module):
    """
    Cross-attention: queries from x (audio), keys/values from cond (text).
    Uses FlashAttention when available and no padding mask is needed; otherwise SDPA.
    key_padding_mask: (B, St) bool where True = valid (kept).
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.attn_drop = attn_drop

        self.q_linear  = nn.Linear(d_model, d_model, bias=False)
        self.k_linear  = nn.Linear(d_model, d_model, bias=False)
        self.v_linear  = nn.Linear(d_model, d_model, bias=False)
        self.proj      = nn.Linear(d_model, d_model, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: Tensor,                        # (B, Sa, D) queries
        cond: Tensor,                     # (B, St, D) keys/values
        key_padding_mask: Optional[Tensor] = None  # (B, St) bool, True=valid
    ) -> Tensor:
        B, Sa, D = x.shape
        St = cond.shape[1]
        H  = self.num_heads
        Hd = self.head_dim

        # Linear projections
        q_lin = self.q_linear(x)          # (B, Sa, D)
        k_lin = self.k_linear(cond)       # (B, St, D)
        v_lin = self.v_linear(cond)       # (B, St, D)

        # SDPA with key padding mask
        q = q_lin.view(B, Sa, H, Hd).transpose(1, 2)  # (B,H,Sa,Hd)
        k = k_lin.view(B, St, H, Hd).transpose(1, 2)  # (B,H,St,Hd)
        v = v_lin.view(B, St, H, Hd).transpose(1, 2)  # (B,H,St,Hd)

        attn_mask = None
        if key_padding_mask is not None:
            # SDPA expects True = mask(disallow); ours True = keep => invert
            attn_mask = (~key_padding_mask)[:, None, None, :]  # (B,1,1,St)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop if self.training else 0.0
        )                               # (B,H,Sa,Hd)
        out = rearrange(out, "b h s d -> b s (h d)")

        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class DDiTBlockCross(nn.Module):
    """
    Keep the original structure with explicit skip connections:
      x -> (SelfAttn) -> skip -> (CrossAttn) -> skip -> (MLP) -> skip
    All modulated by adaLN(time). Reuse gate_msa for both attn residuals (minimal change).
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
        assert dim % n_heads == 0, "dim must be divisible by n_heads"

        self.n_heads = n_heads
        self.dim = dim
        self.dropout = dropout

        # Self-attention + norm
        self.norm1 = LayerNorm(dim=dim)
        self.self_attn = MultiHeadSelfAttention(
            d_model=dim,
            num_heads=n_heads,
            attn_drop=dropout,
            proj_drop=dropout
        )

        # Cross-attention + norm
        self.norm_xattn = LayerNorm(dim=dim)
        self.cross_attn = MultiHeadCrossAttention(
            d_model=dim,
            num_heads=n_heads,
            attn_drop=dropout,
            proj_drop=dropout
        )

        # MLP
        self.norm2 = LayerNorm(dim=dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )

        # adaLN (time)
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(
        self,
        x: Tensor,                   # (B, Sa, D)
        text_emb: Tensor,            # (B, St, D)
        text_att_mask: Optional[Tensor],  # (B, St) bool: True=valid
        rotary_cos_sin: Optional[Tensor],
        c: Tensor
    ) -> Tensor:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

        # ---- Self-attention + skip ----
        x_sa_in = modulate(self.norm1(x), shift=shift_msa, scale=scale_msa)
        sa_out = self.self_attn(x_sa_in, rotary_cos_sin=rotary_cos_sin, attn_mask=None)
        x = bias_dropout_add_scale(
            x=sa_out,
            scale=gate_msa,
            residual=x,
            prob=self.dropout,
            training=self.training,
        )

        # ---- Cross-attention + skip ----
        x_ca_in = modulate(self.norm_xattn(x), shift=shift_msa, scale=scale_msa)
        ca_out = self.cross_attn(x_ca_in, text_emb, key_padding_mask=text_att_mask)
        x = bias_dropout_add_scale(
            x=ca_out,
            scale=gate_msa,
            residual=x,
            prob=self.dropout,
            training=self.training,
        )

        # ---- MLP + skip ----
        x = bias_dropout_add_scale(
            x=self.mlp(modulate(self.norm2(x), shift=shift_mlp, scale=scale_mlp)),
            scale=gate_mlp,
            residual=x,
            prob=self.dropout,
            training=self.training,
        )
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
        x = self.linear(x)
        return x


class TransformerCrossAttn(nn.Module):
    """
    Text is injected via cross-attention (as memory), time via adaLN.
    Self-attn path keeps FlashAttention when available.
    """
    def __init__(
        self,
        audio_vocab_size: int,
        text_vocab_size: int,
        hidden_size: int,
        cond_dim: int,
        n_heads: int,
        dropout: int,
        n_blocks: int,
        add_token: int = 2,
        audio_pad_token: Optional[int] = 0
    ):
        super().__init__()
        self.audio_vocab_size = audio_vocab_size
        self.audio_pad_token = audio_pad_token
        self.text_vocab_size = text_vocab_size

        self.audio_embed = nn.Embedding(self.audio_vocab_size + add_token, hidden_size)
        self.text_embed  = nn.Embedding(self.text_vocab_size + 1, hidden_size)  # +1 for filler=0

        self.time_embedding = TimestepEmbedder(hidden_size=cond_dim)
        self.rotary_emb = rotary.Rotary(dim=hidden_size // n_heads)

        self.input_proj = nn.Linear(hidden_size, hidden_size)

        self.blocks = nn.ModuleList(
            [
                DDiTBlockCross(
                    dim=hidden_size,
                    n_heads=n_heads,
                    cond_dim=cond_dim,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )

        self.output_layer = DDitFinalLayer(
            hidden_size=hidden_size,
            out_channels=audio_vocab_size + add_token,
            cond_dim=cond_dim,
        )

    def forward(
        self,
        x_t: Tensor,               # (B, Sa)
        text: Tensor,              # (B, St) int ids
        text_att_mask: Tensor,     # (B, St) bool: True for valid tokens
        time: Tensor,              # (B,)
        drop_text: bool = False,
    ) -> Tensor:
        # Audio embeddings
        x = self.audio_embed(x_t)          # (B, Sa, D)
        x = self.input_proj(x)             # (B, Sa, D)

        # Text embeddings (+1 shift so 0 is filler)
        text_ids = text + 1
        if drop_text:
            text_ids = torch.zeros_like(text_ids) # Filler tokens to represent "dropped text" for PFG
            text_att_mask = torch.ones_like(text_att_mask, dtype=torch.bool) # This will be inverted to False in the cross attention module
        text_emb = self.text_embed(text_ids)  # (B, St, D)

        # Time conditioning
        c = F.silu(self.time_embedding(time=time))  # (B, cond_dim)

        # Rotary (audio stream) for self-attn
        rotary_cos_sin = self.rotary_emb(x=x)

        # Blocks
        for blk in self.blocks:
            x = blk(
                x=x,
                text_emb=text_emb,
                text_att_mask=text_att_mask,
                rotary_cos_sin=rotary_cos_sin,
                c=c
            )

        # Logits
        x = self.output_layer(x=x, c=c)
        return x
