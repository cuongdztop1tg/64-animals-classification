import torch.nn as nn
from .residual_block import ResidualBlock


class ResNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=input_dim,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
        )

        self.stage1 = self.make_stage(64, 64, 3, 1)
        self.stage2 = self.make_stage(256, 128, 4, 2)
        self.stage3 = self.make_stage(512, 256, 6, 2)
        self.stage4 = self.make_stage(1024, 512, 3, 2)

        self.avgpool2d = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=output_dim),
        )

    def make_stage(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))

        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels * 4, out_channels, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool2d(x)
        x = self.fc(x)

        return x
