from collections import OrderedDict

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch

class _DropoutNd(nn.Module):
    __constants__ = ['p', 'inplace']
    p: float
    inplace: bool

    def __init__(self, p: float = 0.05, inplace: bool = False) -> None:
        super(_DropoutNd, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def extra_repr(self) -> str:
        return 'p={}, inplace={}'.format(self.p, self.inplace)


class Dropout3d_always(_DropoutNd):
    r"""Randomly zero out entire channels (a channel is a 3D feature map,
    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 3D tensor :math:`\text{input}[i, j]`).
    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    Alwyas applies dropout also during evaluation

    As described in the paper
    `Efficient Object Localization Using Convolutional Networks`_ ,
    if adjacent pixels within feature maps are strongly correlated
    (as is normally the case in early convolution layers) then i.i.d. dropout
    will not regularize the activations and will otherwise just result
    in an effective learning rate decrease.

    In this case, :func:`nn.Dropout3d` will help promote independence between
    feature maps and should be used instead.

    Args:
        p (float, optional): probability of an element to be zeroed.
        inplace (bool, optional): If set to ``True``, will do this operation
            in-place

    Shape:
        - Input: :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)`.
        - Output: :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)`
                  (same shape as input).

    Examples::

        >>> m = nn.Dropout3d(p=0.2)
        >>> input = torch.randn(20, 16, 4, 32, 32)
        >>> output = m(input)

    .. _Efficient Object Localization Using Convolutional Networks:
       https://arxiv.org/abs/1411.4280
    """

    def forward(self, input: Tensor) -> Tensor:
        return F.dropout3d(input, self.p, True, self.inplace)
    

class ConvTranspose3dSame(nn.ConvTranspose3d):
    def forward(self, x: torch.Tensor, output_size=None) -> torch.Tensor:
        # Compute output size if not provided
        if output_size is None:
            # Default PyTorch logic, may not match encoder shape exactly
            return super().forward(x)

        # Compute required output_padding
        # Based on the ConvTranspose3d formula:
        # out = (in - 1) * stride - 2 * padding + kernel + output_padding

        # Calculate expected output shape
        in_shape = x.shape[-3:]
        stride = self.stride
        kernel = self.kernel_size
        dilation = self.dilation
        padding = self.padding

        expected_output_shape = [
            (in_shape[i] - 1) * stride[i]
            - 2 * padding[i]
            + dilation[i] * (kernel[i] - 1)
            + 1
            for i in range(3)
        ]

        # Compute needed output_padding
        output_padding = [
            output_size[i] - expected_output_shape[i]
            for i in range(3)
        ]

        return F.conv_transpose3d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=tuple(output_padding),
            dilation=self.dilation,
            groups=self.groups
        )


class DecoderNet(pl.LightningModule):
    r"""TO DO
    """

    def __init__(self,
                 latent_dim=32,
                 output_shape=(1, 64, 64, 64),
                 filters=[128,64,32], 
                 drop_rate=0.05
                 ):

        super().__init__()
        
        self.output_shape = output_shape  # (C, D, H, W)
        c = output_shape[0]
        self.init_channels = filters[0]
   

        self.fc = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(latent_dim, self.init_channels*5*5*3)),
            ("relu1", nn.ReLU()),
        ]))

        modules = []
        in_channels = self.init_channels
        for i in range(len(filters)): 
            out_channels = filters[i]

            modules.append((
                f"ConvTranspose3d{i}",
                nn.ConvTranspose3d(
                    in_channels, out_channels,
                    kernel_size=3, stride=2, padding=1  # doubles the spatial size
                )
            ))
            modules.append((f"bn{i}", nn.BatchNorm3d(out_channels)))
            modules.append((f"lrelu{i}", nn.LeakyReLU(inplace=True)))
            modules.append((f"dropout{i}", nn.Dropout3d(p=drop_rate)))

            in_channels = out_channels

            modules.append((
                f"ConvTranspose3d{i}_2",
                nn.ConvTranspose3d(
                    in_channels, out_channels,
                    kernel_size=3, stride=1, padding=1  
                )
            ))
            modules.append((f"bn{i}", nn.BatchNorm3d(out_channels)))
            modules.append((f"lrelu{i}", nn.LeakyReLU(inplace=True)))
            modules.append((f"dropout{i}", nn.Dropout3d(p=drop_rate)))

            modules.append((
                f"ConvTranspose3d{i}_3",
                nn.ConvTranspose3d(
                    in_channels, out_channels,
                    kernel_size=3, stride=1, padding=1  
                )
            ))
            modules.append((f"bn{i}", nn.BatchNorm3d(out_channels)))
            modules.append((f"lrelu{i}", nn.LeakyReLU(inplace=True)))
            modules.append((f"dropout{i}", nn.Dropout3d(p=drop_rate)))

            modules.append((
                f"ConvTranspose3d{i}_4",
                nn.ConvTranspose3d(
                    in_channels, out_channels,
                    kernel_size=3, stride=1, padding=1  
                )
            ))
            modules.append((f"bn{i}", nn.BatchNorm3d(out_channels)))
            modules.append((f"lrelu{i}", nn.LeakyReLU(inplace=True)))
            modules.append((f"dropout{i}", nn.Dropout3d(p=drop_rate)))

        modules.append((
            "final_conv",
            nn.Conv3d(in_channels, c, kernel_size=7, stride=1, padding=1)
        ))

        self.decoder = nn.Sequential(OrderedDict(modules))

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.init_channels, 5, 5, 3)
        x = self.decoder(x)

        # Resize if needed due to rounding
        if x.shape[2:] != self.output_shape[1:]:
            #print('Output shape are different:', x.shape[2:], 'VS', self.output_shape[1:])
            #print('Trilinear interpolation is used then')
            x = F.interpolate(x, size=self.output_shape[1:], mode='trilinear', align_corners=False)

        return x