from collections import OrderedDict

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


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


class DecoderNet(pl.LightningModule):
    r"""3D-ConvNet model class, based on

    Attributes:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first
            convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate
        num_classes (int) - number of classification classes
            (if 'classifier' mode)
        in_channels (int) - number of input channels (1 for sMRI)
        mode (str) - specify in which mode DenseNet is trained on,
            must be "encoder" or "classifier"
        memory_efficient (bool) - If True, uses checkpointing. Much more memory
            efficient, but slower. Default: *False*.
            See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self,
                 latent_dim=32,
                 output_shape=(1, 64, 64, 64),
                 filters=[128,64,32], 
                 drop_rate=0.05
                 ):

        super().__init__()
        
        self.output_shape = output_shape  # (C, D, H, W)
        c, d, h, w = output_shape
        self.init_d = d // (2 ** len(filters))
        self.init_h = h // (2 ** len(filters))
        self.init_w = w // (2 ** len(filters))
        self.init_channels = filters[0]
        self.volume_size = self.init_d * self.init_h * self.init_w * self.init_channels

        self.fc = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(latent_dim, 512)),
            ("relu1", nn.ReLU()),
            ("fc2", nn.Linear(512, self.volume_size)),
            ("relu2", nn.ReLU())
        ]))

        # Upsampling path
        modules = []
        in_channels = self.init_channels
        for i in range(3):  # 3 blocks for depth
            out_channels = filters[i] if i < len(filters) else filters[-1]

            modules.append((
                f"deconv{i}",
                nn.ConvTranspose3d(
                    in_channels, out_channels,
                    kernel_size=4, stride=2, padding=1  # doubles the spatial size
                )
            ))
            modules.append((f"bn{i}", nn.BatchNorm3d(out_channels)))
            modules.append((f"lrelu{i}", nn.LeakyReLU(inplace=True)))
            modules.append((f"dropout{i}", nn.Dropout3d(p=drop_rate)))

            in_channels = out_channels

        # Final conv to match desired output channels (e.g., 1)
        modules.append((
            "final_conv",
            nn.Conv3d(in_channels, c, kernel_size=3, stride=1, padding=1)
        ))

        self.decoder = nn.Sequential(OrderedDict(modules))

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.init_channels, self.init_d, self.init_h, self.init_w)
        x = self.decoder(x)

        # Resize if needed due to rounding
        if x.shape[2:] != self.output_shape[1:]:
            x = F.interpolate(x, size=self.output_shape[1:], mode='trilinear', align_corners=False)

        return x