"""
Differences from https://github.com/JJGO/UniverSeg/blob/main/universeg/model.py
* Options for different normalization
* Options for different skip connections
* replaced CrossConv with faster version 
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import einops as E
import torch
from torch import nn

from pylot.nn.norm import get_normlayer, NormType

from universeg.nn.init import reset_conv2d_parameters
from universeg.nn.vmap import Vmap, vmap
from universeg.model import get_nonlinearity
from universeg.validation import (
    Kwargs,
    as_2tuple,
    size2t,
    validate_arguments,
    validate_arguments_init,
)

from multiverseg.nn.cross_conv import FastCrossConv2d

@validate_arguments_init
@dataclass(eq=False, repr=False)
class ConvOp(nn.Sequential):

    in_channels: int
    out_channels: int
    kernel_size: size2t = 3
    nonlinearity: Optional[str] = "LeakyReLU"
    norm: Optional[NormType] = None
    init_distribution: Optional[str] = "kaiming_normal"
    init_bias: Union[None, float, int] = 0.0

    def __post_init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            padding_mode="zeros",
            bias=True,
        )

        if self.nonlinearity is not None:
            self.nonlin = get_nonlinearity(self.nonlinearity)

        if self.norm is not None:
            # norm before activation for two reasons:
            # 1. centering before the nonlinearity will maximize the utility of nonlinearity
            # 2. otherwise, centering a only positive distribution will mess with it's shape
            #    (see the Bengio's Deep Learning book)
            self.norml = get_normlayer(self.out_channels, kind=self.norm, dims=2)

        reset_conv2d_parameters(
            self, self.init_distribution, self.init_bias, self.nonlinearity
        )
    


@validate_arguments_init
@dataclass(eq=False, repr=False)
class CrossOp(nn.Module):
    in_channels: size2t
    out_channels: int
    kernel_size: size2t = 3
    nonlinearity: Optional[str] = "LeakyReLU"
    norm: Optional[NormType] = None
    norm_support_dim: Optional[Literal["B", "S"]] = "B" # dimension to roll the support into for normalization
    init_distribution: Optional[str] = "kaiming_normal"
    init_bias: Union[None, float, int] = 0.0

    def __post_init__(self):
        super().__init__()

        self.cross_conv = FastCrossConv2d(
            in_channels=as_2tuple(self.in_channels),
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
        )

        if self.nonlinearity is not None:
            self.nonlin = get_nonlinearity(self.nonlinearity)

        if self.norm is not None:
            assert self.norm_support_dim is not None, "norm_support_dim must be specified. Options are 'B' or 'S'"
            self.norml = get_normlayer(self.out_channels, kind=self.norm, dims=2)

        reset_conv2d_parameters(
            self, self.init_distribution, self.init_bias, self.nonlinearity
        )

    def forward(self, target, support=None):
        interaction = self.cross_conv(target, support).squeeze(dim=1)

        if self.nonlinearity is not None:
            interaction = vmap(self.nonlin, interaction)
       
        if self.norm is not None:
            if self.norm_support_dim == "B":
                # Roll support dimension into batch dimension
                B, S, C, *_ = interaction.shape
                interaction = E.rearrange(interaction, "B S C H W -> (B S) C H W")
                interaction = self.norml(interaction)
                interaction = E.rearrange(interaction, "(B S) C H W -> B S C H W", B=B, S=S, C=C)
            elif self.norm_support_dim == "S":
                interaction = vmap(self.norml, interaction)
            else:
                raise ValueError("norm_support_dim must be 'B' or 'C'")
            
        new_target = interaction.mean(dim=1, keepdims=True)

        return new_target, interaction


class Residual(nn.Module):
    @validate_arguments
    def __init__(
        self,
        module,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.main = module
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            # TODO do we want to init these to 1, like controlnet's zeroconv
            # TODO do we want to initialize these like the other conv layers
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=False
            )
            reset_conv2d_parameters(self.shortcut, "kaiming_normal", 0.0)

    def forward(self, input):
        return self.main(input) + self.shortcut(input)


class VResidual(Residual):
    def forward(self, input):
        return self.main(input) + vmap(self.shortcut, input)


@validate_arguments
def get_shortcut(in_channels: int, out_channels: int) -> nn.Module:
    if in_channels == out_channels:
        shortcut = nn.Identity()
    else:
        shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    return Vmap(shortcut)


@validate_arguments_init
@dataclass(eq=False, repr=False)
class CrossBlock(nn.Module):
    in_channels: size2t
    cross_features: int
    conv_features: Optional[int] = None
    cross_kws: Optional[Dict[str, Any]] = None
    conv_kws: Optional[Dict[str, Any]] = None
    cross_residual: bool = False
    conv_residual: bool = False

    def __post_init__(self):
        super().__init__()

        conv_features = self.conv_features or self.cross_features
        cross_kws = self.cross_kws or {}
        conv_kws = self.conv_kws or {}

        self.cross = CrossOp(self.in_channels, self.cross_features, **cross_kws)
        self.target = Vmap(ConvOp(self.cross_features, conv_features, **conv_kws))
        self.support = Vmap(ConvOp(self.cross_features, conv_features, **conv_kws))

        if self.cross_residual:
            t_ch, sup_ch = as_2tuple(self.cross.in_channels)
            self.target_cross_shortcut = get_shortcut(t_ch, self.cross_features)
            self.support_cross_shortcut = get_shortcut(sup_ch, self.cross_features)
        if self.conv_residual:
            self.target_conv_shortcut = get_shortcut(self.cross_features, conv_features)
            self.support_conv_shortcut = get_shortcut(
                self.cross_features, conv_features
            )

    def forward(self, target, support):
        prev_target, prev_support = target, support
        target, support = self.cross(target, support)
        if self.cross_residual:
            target = target + self.target_cross_shortcut(prev_target)
            support = support + self.support_cross_shortcut(prev_support)

        prev_target, prev_support = target, support

        target = self.target(target)
        support = self.support(support)
        if self.conv_residual:
            target = target + self.target_conv_shortcut(prev_target)
            support = support + self.support_conv_shortcut(prev_support)

        return target, support


@validate_arguments_init
@dataclass(eq=False, repr=False)
class MultiverSegNet(nn.Module):
    encoder_blocks: List[size2t]
    decoder_blocks: Optional[List[size2t]] = None
    cross_relu: bool = True
    block_kws: Optional[Kwargs] = None
    in_channels: Tuple[int, int] = (1, 2)

    def __post_init__(self):
        super().__init__()

        self.downsample = nn.MaxPool2d(2, 2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.enc_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        encoder_blocks = list(map(as_2tuple, self.encoder_blocks))
        decoder_blocks = self.decoder_blocks or encoder_blocks[-2::-1]
        decoder_blocks = list(map(as_2tuple, decoder_blocks))

        block_kws = self.block_kws or {}
        if "cross_kws" not in block_kws:
            block_kws["cross_kws"] = {"nonlinearity": "LeakyReLU"}
        if not self.cross_relu:
            block_kws["cross_kws"]["nonlinearity"] = None

        in_ch = self.in_channels
        out_channels = 1
        out_activation = None

        # Encoder
        skip_outputs = []
        for cross_ch, conv_ch in encoder_blocks:
            block = CrossBlock(in_ch, cross_ch, conv_ch, **block_kws)
            in_ch = conv_ch
            self.enc_blocks.append(block)
            skip_outputs.append(in_ch)

        # Decoder
        skip_chs = skip_outputs[-2::-1]
        for (cross_ch, conv_ch), skip_ch in zip(decoder_blocks, skip_chs):
            block = CrossBlock(in_ch + skip_ch, cross_ch, conv_ch, **block_kws)
            in_ch = conv_ch
            self.dec_blocks.append(block)

        self.out_conv = ConvOp(
            in_ch,
            out_channels,
            kernel_size=1,
            nonlinearity=out_activation,
        )

    def forward(self, target_image, support_images = None, support_labels = None):
        
        # Added for multiverseg -- only add dimension if it's not there
        if len(target_image.shape) == 4:  
            target = E.rearrange(target_image, "B C H W -> B 1 C H W")
        else:
            # ScribblePrompt prompt generator outputs B 1 C H W tensor
            target = target_image

        # Added for multiverseg -- need at least one support image for architecture to work
        if support_images is None or support_images.shape[1]==0:
            bs, _, _, h, w = target.shape
            # Use one support image with all pixels set to 0.5
            support_images = 0.5*torch.ones((bs, 1, 1, h, w), device=target.device)
            support_labels = 0.5*torch.ones((bs, 1, 1, h, w), device=target.device)

        support = torch.cat([support_images, support_labels], dim=2)
        
        pass_through = []

        for i, encoder_block in enumerate(self.enc_blocks):
            target, support = encoder_block(target, support)
            if i == len(self.encoder_blocks) - 1:
                break
            pass_through.append((target, support))
            target = vmap(self.downsample, target)
            support = vmap(self.downsample, support)

        for decoder_block in self.dec_blocks:
            target_skip, support_skip = pass_through.pop()
            target = torch.cat([vmap(self.upsample, target), target_skip], dim=2)
            support = torch.cat([vmap(self.upsample, support), support_skip], dim=2)
            target, support = decoder_block(target, support)

        target = E.rearrange(target, "B 1 C H W -> B C H W")
        target = self.out_conv(target)
        return target


# @validate_arguments
# def universeg(version: Literal["v1"] = "v1", pretrained: bool = False) -> nn.Module:
#     weights = {
#         "v1": "https://github.com/JJGO/UniverSeg/releases/download/weights/universeg_v1_nf64_ss64_STA.pt"
#     }

#     if version == "v1":
#         model = UniverSeg(encoder_blocks=[64, 64, 64, 64])

#     if pretrained:
#         state_dict = torch.hub.load_state_dict_from_url(weights[version])
#         model.load_state_dict(state_dict)

#     return model