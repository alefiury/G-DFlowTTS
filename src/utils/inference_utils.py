from typing import Optional

import torch
from torch import Tensor
from flow_matching.utils import ModelWrapper
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.path.scheduler import PolynomialConvexScheduler

from modules.gdflowtts.flow import MaskedSourceDistribution, UniformSourceDistribution, get_source_distribution


class DFMTTSPosteriorNoCFG(ModelWrapper):
    def __init__(
        self,
        base_model,
        vocab_size: int,
        *,
        prefix_len: int,
        mask_id: int,
        freeze_prefix: bool = True,
        freeze_non_mask: bool = True,
        freeze_eos: bool = True,
        eos_id: int = -1,
        ref_codes: Optional[Tensor] = None
    ):
        super().__init__(base_model)
        self.base_model = base_model
        self.S = int(vocab_size)
        self.prefix_len = int(prefix_len)
        self.mask_id = int(mask_id)
        self.freeze_prefix = bool(freeze_prefix)
        self.freeze_non_mask = bool(freeze_non_mask)
        self.freeze_eos = bool(freeze_eos)
        self.eos_id = int(eos_id)
        self.ref_codes = ref_codes

    @torch.no_grad()
    def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
        """
        Solver calls: self.model(x=x_t, t=t.repeat(B), **extras)
        extras should include: text_ids, text_att_mask, audio_att_mask
        """
        text_ids = extras["text_ids"]
        text_att_mask = extras["text_att_mask"]
        audio_att_mask = extras["audio_att_mask"]

        x_in = x
        if self.ref_codes is not None and self.prefix_len > 0:
            x_in = x.clone()
            x_in[:, :self.prefix_len] = self.ref_codes

        logits = self.base_model(
            x_t=x_in,
            text_ids=text_ids,
            time=t,
            drop_text=False,
            text_att_mask=text_att_mask,
            audio_att_mask=audio_att_mask,
        )

        probs = torch.softmax(logits, dim=-1)

        return probs


@torch.inference_mode()
def sample_with_official_solver(
    *,
    config,
    model,
    text_ids: Tensor,
    text_att_mask: Tensor,
    codes_ref_1d: Tensor,
    suffix_len: int,
    steps: int,
    device: torch.device,
) -> Tensor:
    S = int(config.datasets.audio_vocab_size) + int(config.model.audio_add_token)

    mask_id = int(config.datasets.audio_mask_token)
    eos_id = int(getattr(config.datasets, "audio_eos_token", -1))

    prefix_len = int(codes_ref_1d.numel())
    T = prefix_len + int(suffix_len)

    audio_att_mask = torch.ones((1, T), dtype=torch.bool, device=device)

    source_distribution = get_source_distribution(
        source_distribution=config.source_dist_type,
        mask_token=mask_id,
        vocab_size=S,
    )

    path = MixtureDiscreteProbPath(
        scheduler=PolynomialConvexScheduler(n=1.0)
    )

    wrapped = DFMTTSPosteriorNoCFG(
        base_model=model,
        vocab_size=S,
        prefix_len=prefix_len,
        mask_id=mask_id,
        freeze_prefix=True,
        freeze_non_mask=True,
        freeze_eos=True,
        eos_id=eos_id,
        ref_codes=codes_ref_1d.unsqueeze(0),
    )

    solver = MixtureDiscreteEulerSolver(
        model=wrapped,
        path=path,
        vocabulary_size=S,
        source_distribution_p=None,
    )

    # Vanilla usage: step_size + time_grid=[0,1]
    step_size = 1.0 / int(steps)
    time_grid = torch.tensor([0.0, 1.0])

    x_init = source_distribution.sample(
        tensor_size=(1, T), device=device
    )
    x_init[:, :prefix_len] = codes_ref_1d.unsqueeze(0)
    x_init[:, -1] = eos_id if eos_id >= 0 else mask_id

    x_out = solver.sample(
        x_init=x_init,
        step_size=step_size,
        time_grid=time_grid,
        verbose=False,
        text_ids=text_ids,
        text_att_mask=text_att_mask,
        audio_att_mask=audio_att_mask,
    )
    return x_out