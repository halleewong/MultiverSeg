"""
Copied from https://github.com/JJGO/pylot/blob/master/pylot/nn/norm.py to reduce depencies 
"""
from typing import Literal, Tuple, Union

import einops as E
import torch
from pydantic import validate_arguments
# from pylot.nn import batch_renorm
from torch import nn

NormType = Union[
    Literal["batch", "batchre", "layer", "instance", "channel"],
    Tuple[Literal["group"], int],
]


class ChannelNorm(nn.LayerNorm):
    # Normalizes the C dimension in (N, C, *spatial) tensors
    # This is the LayerNorm in ConvNext Networks

    def __init__(self, num_features, eps=1e-6, affine=True):
        super().__init__(num_features, eps=eps, elementwise_affine=affine)
        self.num_features = num_features

    @property
    def affine(self):
        return self.elementwise_affine

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = E.rearrange(x, "B C ... -> B ... C")
        super().forward(x)
        x = E.rearrange(x, "B ... C -> B C ...")
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_features={self.num_features}, eps={self.eps}, affine={self.affine})"


def get_normlayer(
    features: int, kind: NormType, dims: Literal[1, 2, 3], **norm_kws,
):
    if kind == "batch":
        return getattr(nn, f"BatchNorm{dims}d")(features, **norm_kws)
    # if kind == "batchre":
    #     return getattr(batch_renorm, f"BatchRenorm{dims}d")(features, **norm_kws)
    if kind == "instance":
        # Unlike all others InstanceNorm defaults to affine=False
        # For consistency, we set it to True by default
        norm_kws = {"affine": True, **norm_kws}
        return getattr(nn, f"InstanceNorm{dims}d")(features, **norm_kws)
    if kind == "layer":
        return nn.GroupNorm(1, features, **norm_kws)
    if kind == "channel":
        return ChannelNorm(features, **norm_kws)
    if isinstance(kind, tuple):
        k, groups = kind
        assert k == "group"
        return nn.GroupNorm(groups, features, **norm_kws)