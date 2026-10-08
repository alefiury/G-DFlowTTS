from transformers import PretrainedConfig


class GDFlowTTSConfig(PretrainedConfig):
    """Configuration for the G-DFlowTTS discrete flow matching TTS model.

    The model predicts NeuCodec audio codes (50 Hz) conditioned on GPT-2 text
    tokens, using a masked source distribution and a polynomial convex
    scheduler (alpha_t = t ** scheduler_exponent).
    """

    model_type = "gdflowtts"

    def __init__(
        self,
        audio_vocab_size: int = 65536,
        text_vocab_size: int = 50257,
        hidden_size: int = 768,
        cond_dim: int = 128,
        n_heads: int = 12,
        n_blocks: int = 12,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        audio_add_token: int = 2,
        text_add_token: int = 0,
        text_filler_token: int = 50256,
        audio_eos_token: int = 65536,
        audio_mask_token: int = 65537,
        rotary_base: int = 10_000,
        frequency_embedding_size: int = 256,
        scheduler_exponent: float = 1.0,
        cond_drop_prob: float = 0.0,
        codec_name: str = "neuphonic/neucodec",
        codec_input_sampling_rate: int = 16_000,
        sampling_rate: int = 24_000,
        **kwargs,
    ):
        self.audio_vocab_size = audio_vocab_size
        self.text_vocab_size = text_vocab_size
        self.hidden_size = hidden_size
        self.cond_dim = cond_dim
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.audio_add_token = audio_add_token
        self.text_add_token = text_add_token
        self.text_filler_token = text_filler_token
        self.audio_eos_token = audio_eos_token
        self.audio_mask_token = audio_mask_token
        self.rotary_base = rotary_base
        self.frequency_embedding_size = frequency_embedding_size
        self.scheduler_exponent = scheduler_exponent
        self.cond_drop_prob = cond_drop_prob
        self.codec_name = codec_name
        self.codec_input_sampling_rate = codec_input_sampling_rate
        self.sampling_rate = sampling_rate
        super().__init__(**kwargs)
