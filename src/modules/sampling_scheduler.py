from typing import List, Tuple

import torch
from torch import nn
from einops import rearrange
from torch.nn import functional as F


def x2prob(x: torch.Tensor, dict_size: int) -> torch.Tensor:
    x = F.one_hot(x, num_classes=dict_size)
    return rearrange(x, 'b s c -> b c s')


def sample_p(pt: torch.Tensor) -> torch.Tensor:
    b, _, s = pt.shape
    pt = rearrange(pt, 'b c s -> (b s) c')
    xt = torch.multinomial(pt, 1)
    return xt.reshape(b, s)


class Ccoupling:
    def __init__(
        self,
        mask_prob: float = 0.15,
        special_tokens: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        mask_token: int = 0
    ):
        self.mask_prob = mask_prob
        self.special_tokens = torch.tensor(special_tokens)
        self.mask_token = mask_token

    def sample(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        x1_source, x1_target = batch
        mask = torch.rand_like(x1_source.float()) < self.mask_prob
        # dont mask special characters
        mask = mask & ~torch.isin(x1_source.to(mask.device),  self.special_tokens.to(mask.device)).to(mask.device)
        x0_source = torch.where(mask, torch.full_like(x1_source, self.mask_token), x1_source)
        return (x0_source, x1_target), (x1_source, x1_target)


    def simple_sample(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.rand_like(x.float()) < self.mask_prob
        # dont mask special characters
        mask = mask & ~torch.isin(x.to(mask.device),  self.special_tokens.to(mask.device)).to(mask.device)
        x0 = torch.where(mask, torch.full_like(x, self.mask_token), x)
        return x0


# Kappa Scheduler
class KappaScheduler:
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def derivative(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class CubicScheduler(KappaScheduler):
    def __init__(self, a: float = 2.0, b: float = 0.5):
        self.a = a
        self.b = b

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return -2 * (t**3) + 3 * (t**2) + self.a * (t**3 - 2*t**2 + t) + self.b * (t**3 - t**2) # Equation 33

    def derivative(self, t: torch.Tensor) -> torch.Tensor:
        return -6 * (t**2) + 6 * t + self.a * (3*t**2 - 4*t + 1) + self.b * (3*t**2 - 2*t)