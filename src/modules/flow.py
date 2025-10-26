from abc import ABC
from typing import Optional, Tuple

import torch
from torch import Tensor

from flow_matching.path.scheduler.scheduler import SchedulerOutput, ConvexScheduler


class KOConvexScheduler(ConvexScheduler):
    """KO Scheduler."""

    def __call__(self, t: Tensor) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t=torch.cos(0.5 * torch.pi * (1 - t)) ** 2,
            sigma_t=torch.sin(0.5 * torch.pi * (1 - t)) ** 2,
            d_alpha_t=0.5 * torch.pi * torch.sin(torch.pi * (1 - t)),
            d_sigma_t=0 - 0.5 * torch.pi * torch.sin(torch.pi * (1 - t)),
        )

    def kappa_inverse(self, kappa: Tensor) -> Tensor:
        return 1 - torch.arccos(torch.sqrt(kappa)) * 0.5 / torch.pi


class SourceDistribution(ABC):
    def __init__(
        self,
    ) -> None:
        ...

    def sample(self, tensor_size: Tuple[int, ...], device: torch.device) -> Tensor:
        ...

    def sample_like(self, tensor_like: Tensor) -> Tensor:
        ...


class MaskedSourceDistribution(SourceDistribution):
    def __init__(self, mask_token: int) -> None:
        self.mask_token = mask_token

    @property
    def masked(self) -> bool:
        return True

    def sample(self, tensor_size: Tuple[int, ...], device: torch.device) -> Tensor:
        return torch.zeros(tensor_size, device=device).fill_(self.mask_token).long()

    def sample_like(self, tensor_like: Tensor) -> Tensor:
        return torch.zeros_like(tensor_like).fill_(self.mask_token).long()


class UniformSourceDistribution(SourceDistribution):
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    @property
    def masked(self) -> bool:
        return False

    def sample(self, tensor_size: Tuple[int, ...], device: torch.device) -> Tensor:
        return torch.randint(size=tensor_size, high=self.vocab_size, device=device)

    def sample_like(self, tensor_like: Tensor) -> Tensor:
        return torch.randint_like(tensor_like, high=self.vocab_size)