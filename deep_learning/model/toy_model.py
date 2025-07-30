# toy_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ToyDecoderModel(nn.Module):
    def __init__(self, latent_dim=32, output_shape=(1, 16, 37, 37)):
        super(ToyDecoderModel, self).__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape  # (C, D, H, W)

        self.D, self.H, self.W = output_shape[1:]  # 16, 37, 37

        # Project latent to small 3D feature volume
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 16 * 2 * 5 * 5)  # Output: (B, 16, 2, 5, 5)

        # Decoder network
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(16, 32, kernel_size=3, stride=(2, 2, 2), padding=1, output_padding=(1, 0, 0)),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(32, 32, kernel_size=3, stride=(2, 2, 2), padding=1, output_padding=(1, 1, 1)),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv3d(16, output_shape[0], kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 16, 2, 5, 5)  # reshape to 3D volume
        x = self.decoder(x)

        # Resize to match exact output shape (e.g., 1×16×37×37)
        x = F.interpolate(x, size=self.output_shape[1:], mode='trilinear', align_corners=False)
        return x