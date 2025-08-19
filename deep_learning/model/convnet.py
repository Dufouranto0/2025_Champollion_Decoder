import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import Tensor
from collections import OrderedDict

# --------- Utility functions ---------
def ComputeOutputDim(dimension, depth):
    """Compute the resolution after depth levels of stride-2 downsampling"""
    if depth == 0:
        return dimension
    else:
        return ComputeOutputDim(dimension // 2 + dimension % 2, depth - 1)

def compute_decoder_shapes(output_shape, depth):
    """
    Given the final output shape (D, H, W), compute intermediate shapes
    matching each encoder depth level.
    """
    shapes = [output_shape]
    for _ in range(depth):
        d, h, w = shapes[0]
        shapes.insert(0, (
            ComputeOutputDim(d, 1),
            ComputeOutputDim(h, 1),
            ComputeOutputDim(w, 1)
        ))
    return shapes

# --------- Dropout always ---------
class _DropoutNd(nn.Module):
    def __init__(self, p=0.05, inplace=False):
        super().__init__()
        self.p = p
        self.inplace = inplace

class Dropout3d_always(_DropoutNd):
    def forward(self, x: Tensor) -> Tensor:
        return F.dropout3d(x, self.p, training=True, inplace=self.inplace)

# --------- ConvTranspose3d that matches exact output shape ---------
class ConvTranspose3dSame(nn.ConvTranspose3d):
    def forward(self, x: torch.Tensor, output_size=None) -> torch.Tensor:
        if output_size is None:
            return super().forward(x)

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

# --------- Main Decoder ---------
class DecoderNet(pl.LightningModule):
    def __init__(self,
                 latent_dim=32,
                 output_shape=(1, 37, 37, 16),
                 filters=[128, 64, 32],
                 drop_rate=0.05,
                 loss_name="ce"):

        super().__init__()

        self.output_shape = output_shape
        self.loss_name = loss_name.lower()
        self.init_channels = filters[0]

        # Compute spatial shapes from deepest to output
        self.spatial_shapes = compute_decoder_shapes(self.output_shape[1:], len(filters))

        # FC layer projects latent to spatial volume
        init_shape = self.spatial_shapes[0]
        self.fc1 = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(latent_dim, 
                              self.init_channels * init_shape[0] * init_shape[1] * init_shape[2])),
            ("BN1d", nn.BatchNorm1d(self.init_channels * init_shape[0] * init_shape[1] * init_shape[2])),
            ("relu", nn.ReLU())
        ]))

        self.blocks = nn.ModuleList()
        self.target_shapes = []

        in_channels = self.init_channels
        for i, out_channels in enumerate(filters):
            target_shape = self.spatial_shapes[i + 1]

            block = nn.ModuleList([
                ConvTranspose3dSame(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU(inplace=True),
                Dropout3d_always(p=drop_rate),
                
                # ConvTranspose3d or Conv3d ??
                nn.ConvTranspose3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU(inplace=True),
                Dropout3d_always(p=drop_rate),

            ])
            self.blocks.append(block)
            self.target_shapes.append(target_shape)
            in_channels = out_channels

        # Final layer adapts to loss type
        if self.loss_name == "mse":
            self.final_conv = nn.Conv3d(in_channels, self.output_shape[0], kernel_size=3, stride=1, padding=1)
        elif self.loss_name == "bce":
            self.final_conv = nn.Conv3d(in_channels, self.output_shape[0], kernel_size=1, stride=1)
        elif self.loss_name == "ce":
            self.final_conv = nn.Conv3d(in_channels, self.output_shape[0], kernel_size=1, stride=1)
        else:
            raise ValueError(f"Unsupported loss type: {loss_name}")

    def forward(self, x):
        d, h, w = self.spatial_shapes[0]
        x = self.fc1(x)
        x = x.view(-1, self.init_channels, d, h, w)

        for block, target_shape in zip(self.blocks, self.target_shapes):
            x = block[0](x, output_size=target_shape)
            for layer in block[1:]:
                x = layer(x)

        return self.final_conv(x)

