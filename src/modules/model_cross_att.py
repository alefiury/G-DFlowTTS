import math
from typing import Optional, Literal

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange

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


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return (self.weight * self._norm(x.float())).type_as(x)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
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
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, time: Tensor) -> Tensor:
        t_freq = self.timestep_embedding(time=time, dim=self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LinearAttention(nn.Module):
    """SANA-style Linear Attention"""
    def __init__(self, dim: int, n_heads: int, qk_norm: bool = True):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        if qk_norm:
            self.q_norm = RMSNorm(dim)
            self.k_norm = RMSNorm(dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x: Tensor, rotary_cos_sin: Optional[Tensor] = None) -> Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, C)
        q, k, v = qkv.unbind(2)

        # Apply normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Reshape for multi-head attention
        q = q.reshape(B, N, self.n_heads, self.head_dim)
        k = k.reshape(B, N, self.n_heads, self.head_dim)
        v = v.reshape(B, N, self.n_heads, self.head_dim)

        # Apply rotary embeddings if provided
        if rotary_cos_sin is not None:
            cos, sin = rotary_cos_sin
            q = rotary.apply_rotary_emb_torch(
                x=q.float(),
                cos=cos.float(),
                sin=sin.float()
            ).to(q.dtype)
            k = rotary.apply_rotary_emb_torch(
                x=k.float(),
                cos=cos.float(),
                sin=sin.float()
            ).to(k.dtype)

        # Transpose for attention computation
        q = q.transpose(1, 2)  # (B, H, N, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Linear attention with ReLU
        q = F.relu(q)
        k = F.relu(k)

        # Compute shared terms for efficiency
        kv = torch.matmul(k.transpose(-2, -1), v)  # (B, H, D, D)
        k_sum = k.sum(dim=-2, keepdim=True)  # (B, H, 1, D)

        # Compute attention output
        out = torch.matmul(q, kv) / (torch.matmul(q, k_sum.transpose(-2, -1)) + 1e-6)

        # Reshape back
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)

        return out


class CrossAttention(nn.Module):
    """Multi-head Cross Attention"""
    def __init__(
        self,
        dim: int,
        n_heads: int,
        context_dim: Optional[int] = None,
        qk_norm: bool = False,
        attention_type: Literal["vanilla", "linear"] = "vanilla"
    ):
        super().__init__()
        assert dim % n_heads == 0

        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.context_dim = context_dim or dim
        self.attention_type = attention_type

        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(self.context_dim, dim * 2, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        if qk_norm:
            self.q_norm = RMSNorm(dim)
            self.k_norm = RMSNorm(dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        B, N, C = x.shape
        _, M, _ = context.shape

        q = self.q(x)
        kv = self.kv(context).reshape(B, M, 2, C)
        k, v = kv.unbind(2)

        # Apply normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Reshape for multi-head attention
        q = q.reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, M, self.n_heads, self.head_dim).transpose(1, 2)

        if self.attention_type == "vanilla":
            # Standard scaled dot-product attention
            out = F.scaled_dot_product_attention(q, k, v)
        else:  # linear
            # SANA-style linear cross-attention
            q = F.relu(q)
            k = F.relu(k)

            # Compute shared terms
            kv = torch.matmul(k.transpose(-2, -1), v)  # (B, H, D, D)
            k_sum = k.sum(dim=-2, keepdim=True)  # (B, H, 1, D)

            # Compute attention output
            out = torch.matmul(q, kv) / (torch.matmul(q, k_sum.transpose(-2, -1)) + 1e-6)

        # Reshape back
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)

        return out


class MixFFN(nn.Module):
    """SANA-style Mix-FFN with depth-wise convolution"""
    def __init__(self, dim: int, mlp_ratio: int = 4):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)

        # Inverted residual block with GLU
        self.fc1 = nn.Linear(dim, hidden_dim * 2, bias=True)
        self.dwconv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3,
                                padding=1, groups=hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape

        # GLU activation
        x_gate = self.fc1(x)
        x, gate = x_gate.chunk(2, dim=-1)
        x = x * F.silu(gate)

        # Apply depth-wise convolution
        x = x.transpose(1, 2)  # (B, C, N)
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, N, C)

        # Final projection
        x = self.fc2(x)

        return x


class CrossDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        context_dim: int,
        cond_dim: int,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        use_linear_attn: bool = False,
        use_mix_ffn: bool = False,
        qk_norm: bool = True,
        cross_qk_norm: bool = True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.dropout = dropout
        self.use_linear_attn = use_linear_attn

        # Self-attention
        self.norm1 = LayerNorm(dim=dim)
        if use_linear_attn:
            self.self_attn = LinearAttention(dim, n_heads, qk_norm=qk_norm)
        else:
            # Vanilla self-attention
            self.qw = nn.Linear(dim, dim, bias=False)
            self.kw = nn.Linear(dim, dim, bias=False)
            self.vw = nn.Linear(dim, dim, bias=False)
            self.attn_out = nn.Linear(dim, dim, bias=False)

            if qk_norm:
                self.q_norm = LayerNorm(dim)
                self.k_norm = LayerNorm(dim)
            else:
                self.q_norm = nn.Identity()
                self.k_norm = nn.Identity()

        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention
        self.norm_cross = LayerNorm(dim=dim)
        self.cross_attn = CrossAttention(
            dim,
            n_heads,
            context_dim=context_dim,
            qk_norm=cross_qk_norm,
            attention_type="linear" if use_linear_attn else "vanilla"
        )
        self.dropout_cross = nn.Dropout(dropout)

        # Feed-forward
        self.norm2 = LayerNorm(dim=dim)
        if use_mix_ffn:
            self.mlp = MixFFN(dim, mlp_ratio)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(dim, mlp_ratio * dim, bias=True),
                nn.GELU(approximate="tanh"),
                nn.Linear(mlp_ratio * dim, dim, bias=True),
            )

        # AdaLN modulation - we need 9 parameters for cross-attention
        self.adaLN_modulation = nn.Linear(cond_dim, 9 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

        self.head_dim = self.dim // self.n_heads

    def forward(
        self,
        x: Tensor,
        context: Tensor,
        rotary_cos_sin: Tensor,
        c: Tensor
    ) -> Tensor:
        batch_size, seq_len = x.shape[0], x.shape[1]

        # Get modulation parameters - expecting 9 values
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_cross,
            scale_cross,
            gate_cross,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c)[:, None].chunk(9, dim=2)

        # Self-attention
        x_skip = x
        x_norm = modulate(x=self.norm1(x), shift=shift_msa, scale=scale_msa)

        if self.use_linear_attn:
            x_attn = self.self_attn(x_norm, rotary_cos_sin)
        else:
            # Vanilla self-attention path
            q = self.qw(x_norm)
            k = self.kw(x_norm)
            v = self.vw(x_norm)

            q = self.q_norm(q)
            k = self.k_norm(k)

            q, k, v = (
                item.view(batch_size, seq_len, self.n_heads, self.head_dim)
                for item in (q, k, v)
            )

            with torch.amp.autocast("cuda", enabled=False):
                cos, sin = rotary_cos_sin
                original_dtype = q.dtype

                q = rotary.apply_rotary_emb_torch(
                    x=q.float(),
                    cos=cos.float(),
                    sin=sin.float()
                ).to(original_dtype)
                k = rotary.apply_rotary_emb_torch(
                    x=k.float(),
                    cos=cos.float(),
                    sin=sin.float()
                ).to(original_dtype)

            q, k, v = (item.transpose(1, 2) for item in (q, k, v))

            x_attn = F.scaled_dot_product_attention(query=q, key=k, value=v)
            x_attn = rearrange(x_attn, "b h s d -> b s (h d)", b=batch_size)
            x_attn = self.attn_out(x_attn)

        x = bias_dropout_add_scale(
            x=x_attn,
            scale=gate_msa,
            residual=x_skip,
            prob=self.dropout,
            training=self.training,
        )

        # Cross-attention
        x_skip = x
        x_norm = modulate(x=self.norm_cross(x), shift=shift_cross, scale=scale_cross)
        x_cross = self.cross_attn(x_norm, context)
        x = bias_dropout_add_scale(
            x=x_cross,
            scale=gate_cross,
            residual=x_skip,
            prob=self.dropout,
            training=self.training,
        )

        # Feed-forward
        x = bias_dropout_add_scale(
            x=self.mlp(modulate(x=self.norm2(x), shift=shift_mlp, scale=scale_mlp)),
            scale=gate_mlp,
            residual=x,
            prob=self.dropout,
            training=self.training,
        )

        return x


class CrossDitFinalLayer(nn.Module):
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
        x = modulate(x=self.norm_final(x), shift=shift, scale=scale)
        x = self.linear(x)
        return x


class CrossTransformer(nn.Module):
    def __init__(
        self,
        audio_vocab_size: int,
        text_vocab_size: int,
        hidden_size: int,
        cond_dim: int,
        n_heads: int,
        dropout: int,
        n_blocks: int,
        add_token: int = 2,  # mask + padding tokens
        audio_pad_token: Optional[int] = 0,
        use_linear_attn: bool = False,  # Use SANA linear attention
        use_mix_ffn: bool = False,  # Use SANA Mix-FFN
        qk_norm: bool = True,  # QK normalization for self-attention
        cross_qk_norm: bool = True,  # QK normalization for cross-attention
        mlp_ratio: int = 4,
    ):
        super().__init__()
        self.audio_vocab_size = audio_vocab_size
        self.audio_pad_token = audio_pad_token
        self.text_vocab_size = text_vocab_size
        self.use_linear_attn = use_linear_attn

        # Embeddings
        self.audio_embed = nn.Embedding(self.audio_vocab_size + add_token, hidden_size)
        self.text_embed = nn.Embedding(self.text_vocab_size + 1, hidden_size)

        # Time embedding
        self.time_embedding = TimestepEmbedder(hidden_size=cond_dim)

        # Rotary embeddings (only used for self-attention)
        self.rotary_emb = rotary.Rotary(dim=hidden_size // n_heads)

        # Transformer blocks with cross-attention
        self.blocks = nn.ModuleList(
            [
                CrossDiTBlock(
                    dim=hidden_size,
                    n_heads=n_heads,
                    context_dim=hidden_size * 2,  # text + cond concatenated
                    cond_dim=cond_dim,
                    dropout=dropout,
                    mlp_ratio=mlp_ratio,
                    use_linear_attn=use_linear_attn,
                    use_mix_ffn=use_mix_ffn,
                    qk_norm=qk_norm,
                    cross_qk_norm=cross_qk_norm,
                )
                for _ in range(n_blocks)
            ]
        )

        # Output layer
        self.output_layer = CrossDitFinalLayer(
            hidden_size=hidden_size,
            out_channels=audio_vocab_size + add_token,
            cond_dim=cond_dim,
        )

    def forward(
        self,
        x_t: Tensor,
        text: Tensor,
        cond: Tensor,
        time: Tensor,
        drop_text: bool = False,
        drop_cond: bool = False,
    ) -> Tensor:
        # Audio embedding
        audio_emb = self.audio_embed(x_t)
        seq_len = audio_emb.shape[1]

        # Text Embedding
        text = text + 1  # use 0 as filler token
        text = text[:, :seq_len]
        text = F.pad(text, (0, seq_len - text.shape[1]), value=0.0)

        # Classifier Free Guidance (CFG) for text
        if drop_text:
            text = torch.zeros_like(text)
        text_emb = self.text_embed(text)

        # Classifier Free Guidance (CFG) for condition
        if drop_cond:
            cond = torch.ones_like(cond) * self.audio_pad_token
        cond_emb = self.audio_embed(cond)

        # Create context by concatenating text and condition embeddings
        context = torch.cat([text_emb, cond_emb], dim=-1)

        # Time conditioning
        c = F.silu(self.time_embedding(time=time))

        # Get rotary embeddings for self-attention
        rotary_cos_sin = self.rotary_emb(x=audio_emb)

        # Process through transformer blocks
        x = audio_emb
        for block in self.blocks:
            x = block(x=x, context=context, rotary_cos_sin=rotary_cos_sin, c=c)

        # Final output
        x = self.output_layer(x=x, c=c)

        return x