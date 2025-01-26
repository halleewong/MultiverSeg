"""
Improved CrossConv implementation
"""
import torch
import torch.nn as nn
import einops as E

from typing import Union, Tuple, Optional
from pydantic import validate_arguments

from pylot.util.shapecheck import ShapeChecker
import torch.nn.functional as F

class FastCrossConv2d(nn.Conv2d):
    """
    Compute pairwise convolution between all element of x and all elements of y.
    x, y are tensors of size B,_,C,H,W where _ could be different number of elements in x and y
    essentially, we do a meshgrid of the elements to get B,Sx,Sy,C,H,W tensors, and then
    pairwise conv.
    Args:
        x (tensor): B,Sx,Cx,H,W
        y (tensor): B,Sy,Cy,H,W
    Returns:
        tensor: B,Sx,Sy,Cout,H,W
    """
    """
    CrossConv2d is a convolutional layer that performs pairwise convolutions between elements of two input tensors.

    Parameters
    ----------
    in_channels : int or tuple of ints
        Number of channels in the input tensor(s).
        If the tensors have different number of channels, in_channels must be a tuple
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple of ints
        Size of the convolutional kernel.
    stride : int or tuple of ints, optional
        Stride of the convolution. Default is 1.
    padding : int or tuple of ints, optional
        Zero-padding added to both sides of the input. Default is 0.
    dilation : int or tuple of ints, optional
        Spacing between kernel elements. Default is 1.
    groups : int, optional
        Number of blocked connections from input channels to output channels. Default is 1.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default is True.
    padding_mode : str, optional
        Padding mode. Default is "zeros".
    device : str, optional
        Device on which to allocate the tensor. Default is None.
    dtype : torch.dtype, optional
        Data type assigned to the tensor. Default is None.

    Returns
    -------
    torch.Tensor
        Tensor resulting from the pairwise convolution between the elements of x and y.

    Notes
    -----
    x and y are tensors of size (B, Sx, Cx, H, W) and (B, Sy, Cy, H, W), respectively,
    The function does the cartesian product of the elements of x and y to obtain a tensor
    of size (B, Sx, Sy, Cx + Cy, H, W), and then performs the same convolution for all 
    (B, Sx, Sy) in the batch dimension. Runtime and memory are O(Sx * Sy).

    Examples
    --------
    >>> x = torch.randn(2, 3, 4, 32, 32)
    >>> y = torch.randn(2, 5, 6, 32, 32)
    >>> conv = CrossConv2d(in_channels=(4, 6), out_channels=7, kernel_size=3, padding=1)
    >>> output = conv(x, y)
    >>> output.shape  #(2, 3, 5, 7, 32, 32)
    """
    @validate_arguments
    def __init__(
        self,
        in_channels: Union[int, tuple[int, int]],
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Union[int, str, Tuple[int, ...]] = 0,
        dilation: Union[int, Tuple[int, ...]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        if isinstance(in_channels, int):
            self.in_channels_x = self.in_channels_y = in_channels
        else:
            self.in_channels_x, self.in_channels_y = in_channels

        assert padding_mode == "zeros", "Only zero-padding is supported"

        super().__init__(
            in_channels=self.in_channels_x + self.in_channels_y,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """
        For universeg, x is target and y is support. 
        """
        B, Sx, *_ = x.shape
        B, Sy, *_ = y.shape

        xs = E.rearrange(x, "B 1 Cx H W -> B Cx H W")
        ys = E.rearrange(y, "B Sy Cy H W -> (B Sy) Cy H W")

        wx = self.weight[:,:self.in_channels_x]
        wy = self.weight[:,self.in_channels_x:]

        ox = F.conv2d(xs, wx, 
                      bias=self.bias, 
                      stride=self.stride, 
                      padding=self.padding, 
                      dilation=self.dilation, 
                      groups=self.groups
                      )
        oy = F.conv2d(ys, wy, 
                      bias=None, 
                      stride=self.stride, 
                      padding=self.padding, 
                      dilation=self.dilation, 
                      groups=self.groups
                      )
        
        if B>1:
            if Sy>1:
                ox = torch.repeat_interleave(ox, Sy, dim=0)
            if Sx>1:
                oy = torch.repeat_interleave(oy, Sx, dim=0)

        oz = ox + oy

        output = E.rearrange(
            oz, "(B Sx Sy) Co H W -> B Sx Sy Co H W", B=B, Sx=Sx, Sy=Sy
        )

        return output

