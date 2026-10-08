import math
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn, Tensor
from flash_attn import flash_attn_func

try:
    from flash_attn import flash_attn_func
    _flash_attn_available = True
except ImportError:
    _flash_attn_available = False

from modules.gdflowtts import rotary


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
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
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


class DDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        cond_dim: int,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert dim % n_heads == 0, "dim must be devisable by n_heads"

        self.n_heads = n_heads
        self.dim = dim
        self.dropout = dropout

        self.head_dim = self.dim // self.n_heads

        self.norm1 = LayerNorm(dim=dim)

        self.qw = nn.Linear(dim, dim, bias=False)
        self.kw = nn.Linear(dim, dim, bias=False)
        self.vw = nn.Linear(dim, dim, bias=False)

        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim=dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x: Tensor, rotary_cos_sin: Tensor, c: Tensor, att_mask: Tensor) -> Tensor:
        batch_size, seq_len = x.shape[0], x.shape[1]

        # Makes sure attn_mask has shape (batch_size, 1, 1, seq_len) for broadcasting
        if att_mask.dim() == 2:
            att_mask = att_mask[:, None, None, :]
            att_mask = att_mask.to(device=x.device, dtype=torch.bool)

        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

        x_skip = x
        x = modulate(x=self.norm1(x), shift=shift_msa, scale=scale_msa)

        q = self.qw(x)
        k = self.kw(x)
        v = self.vw(x)

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
        x = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=att_mask
        )
        x = rearrange(x, "b h s d -> b s (h d)", b=batch_size)

        x = bias_dropout_add_scale(
            x=self.attn_out(x),
            scale=gate_msa,
            residual=x_skip,
            prob=self.dropout,
            training=self.training,
        )
        x = bias_dropout_add_scale(
            x=self.mlp(modulate(x=self.norm2(x), shift=shift_mlp, scale=scale_mlp)),
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
        x = modulate(x=self.norm_final(x), shift=shift, scale=scale)
        x = self.linear(x)

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        audio_vocab_size: int,
        text_vocab_size: int,
        hidden_size: int,
        cond_dim: int,
        n_heads: int,
        dropout: int,
        n_blocks: int,
        audio_add_token: int, # mask + padding tokens
        text_add_token: int, # filler token
        audio_pad_token: Optional[int] = None, # padding token
        text_pad_token: Optional[int] = None, # padding token
        text_filler_token: Optional[int] = None, # filler token
        **kwargs,
    ):
        super().__init__()
        true_audio_vocab_size = audio_vocab_size + audio_add_token
        true_text_vocab_size = text_vocab_size + text_add_token

        self.text_vocab_size = text_vocab_size
        self.text_pad_token = text_pad_token
        self.text_filler_token = text_filler_token

        # + audio_add_token to account for the mask and padding tokens
        self.audio_embed = nn.Embedding(true_audio_vocab_size, hidden_size)
        # + 1 to account for the filler token
        self.text_embed = nn.Embedding(true_text_vocab_size, hidden_size)

        self.time_embedding = TimestepEmbedder(hidden_size=cond_dim)
        self.rotary_emb = rotary.Rotary(dim=hidden_size // n_heads)

        # Conditioning dimension
        cond_channels = 2
        # Project concatenated audio and text embeddings back to hidden_size
        # * (2) because we concatenate audio and text embeddings (optional)
        self.input_proj = nn.Linear(hidden_size * cond_channels, hidden_size)

        self.blocks = nn.ModuleList(
            [
                DDiTBlock(
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
            out_channels=true_audio_vocab_size,
            cond_dim=cond_dim,
        )

    def forward(
        self,
        x_t: Tensor,
        text: Tensor,
        time: Tensor,
        drop_text: bool = False,
        text_att_mask: Optional[Tensor] = None,
        audio_att_mask: Optional[Tensor] = None,
    ) -> Tensor:
        audio_emb = self.audio_embed(x_t)
        seq_len = audio_emb.shape[1]

        if audio_att_mask is None:
            audio_att_mask = torch.ones((x_t.shape[0], seq_len), device=x_t.device, dtype=torch.bool)

        if text_att_mask is None:
            text_att_mask = torch.ones((text.shape[0], text.shape[1]), device=text.device, dtype=torch.bool)

        # Text Embedding
        text = text[:, :seq_len]
        text = F.pad(text, (0, seq_len - text.shape[1]), value=self.text_filler_token)  # pad to audio length

        # Classifier Free Guidance (CFG) for the text conditioning
        if drop_text:
            text = torch.full_like(text, self.text_filler_token)
        text_emb = self.text_embed(text)

        x = torch.cat([audio_emb, text_emb], dim=-1)
        # Project concatenated audio and text embeddings back to hidden_size
        x = self.input_proj(x)

        c = F.silu(self.time_embedding(time=time))

        rotary_cos_sin = self.rotary_emb(x=x)

        for i in range(len(self.blocks)):
            x = self.blocks[i](x=x, rotary_cos_sin=rotary_cos_sin, c=c, att_mask=audio_att_mask)

        x = self.output_layer(x=x, c=c)

        return x