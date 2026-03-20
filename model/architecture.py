import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSepConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, stride=stride,
            padding=1, groups=in_channels
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class AudioCNN(nn.Module):
    def __init__(self, num_classes=35):
        super().__init__()

        self.input_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.block1 = nn.Sequential(
            DepthwiseSepConv(32, 64, stride=1),
            nn.MaxPool2d(2)
        )

        self.block2 = nn.Sequential(
            DepthwiseSepConv(64, 128, stride=1),
            nn.MaxPool2d(2)
        )

        self.block3 = DepthwiseSepConv(128, 256, stride=1)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.input_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = AudioCNN(num_classes=35)
    dummy_input = torch.randn(1, 1, 64, 101)
    output = model(dummy_input)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"input shape:  {dummy_input.shape}")
    print(f"output shape: {output.shape}")
    print(f"total params: {total_params:,}")